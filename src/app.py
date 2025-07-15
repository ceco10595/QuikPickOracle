# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
import re
import csv
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

import streamlit as st
from chromadb import PersistentClient
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from llama_cpp import Llama

from prompt_templates import SYSTEM_TEMPLATE, build_prompt
from feedback_db import save as save_feedback          
from feedback_db import _append_positive, _append_negative  
from rapidfuzz import fuzz, process 
from streamlit_feedback import streamlit_feedback
st.session_state.setdefault("to_log", [])   # list of ('pos'|'neg', payload)
st.session_state.setdefault("assistant_meta", {})   # mid → {"q":…, "a":…}
st.session_state.setdefault("pending_q", None)
st.session_state.setdefault("is_thinking", False)
logo = Image.open("images/QuikPick.png")

# set the small icon in the browser tab / window
st.set_page_config(
    page_title="QuikPick Oracle",
    page_icon="images/QuikPick.png",  
    layout="centered",
)

# instead of st.image(..., use_column_width=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("images/QuikPick.png", width=300)   # adjust width to taste


# ── CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH = "models/llama-3-13b-Instruct-Q4_K_M.gguf"
LORA_PATH  = "models/lora-adapter"
VECTOR_DIR = "vectorstore"
ERROR_RE   = re.compile(r"^\d+_\d+$")
MAX_TOKENS = 256
MEM_TURNS  = 8

# ── CACHES ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LLaMA…")
def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        lora_adapter=LORA_PATH if Path(LORA_PATH).exists() else None,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )

@st.cache_resource(show_spinner="Opening vector store…")
def load_store():
    return PersistentClient(path=VECTOR_DIR).get_collection("errors")

@st.cache_resource(show_spinner="Loading embedder…")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")

llm      = load_llm()
store    = load_store()
embedder = load_embedder()

# ── HELPERS ─────────────────────────────────────────────────────────────────
# helpers.py  
def click_fup(q: str):
    st.session_state.next_q = q


def count_similar_questions(q: str, questions: List[str], threshold: int = 90) -> int:
    """
    Returns how many strings in `questions` have a token_set_ratio ≥ threshold
    when compared to q.
    """
    return sum(
        1
        for prev in questions
        if fuzz.token_set_ratio(q, prev) >= threshold
    )

QA_PATH = Path("data/sample_qa.csv")

def append_qa_if_new(code: str, question: str, answer: str) -> None:
    """Append (code, question, answer) to sample_qa.csv unless it exists."""
    QA_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = QA_PATH.exists()
    answer = answer.replace("\n", r"\n")                    
    # read existing rows to avoid duplicates
    seen = set()
    if exists:
        with QA_PATH.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    seen.add(tuple(row[:3]))

    row = (code.strip(), question.strip(), answer.strip())
    if row not in seen:
        with QA_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            # write header once if the file was just created
            if not exists:
                writer.writerow(["ErrorCode", "Question", "Answer"])
            writer.writerow(row)

def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

@st.cache_resource(show_spinner="Loading docs…")
def docs_for_code(code: str) -> List[Dict[str, Any]]:
    res = store.get(
        where={"ErrorCode": code},
        include=["documents", "metadatas", "embeddings"],
    )
    docs = []
    for d, m, e in zip(res["documents"], res["metadatas"], res["embeddings"]):
        if e is None:
            e = embedder.encode([d])[0]
        docs.append({"content": d, "meta": m, "embedding": e})
    return docs

def retrieve_similar(q: str, docs: list[dict], k: int = 3) -> list[dict]:
    # 1  fuzzy QA match first
    qa_hit = best_qa_match(q, docs, min_score=90)
    if qa_hit:
        return [qa_hit]                       # give the “gold” answer only

    # 2 otherwise fall back to embedding similarity
    q_vec = embedder.encode([q])[0]
    scored = sorted(
        docs,
        key=lambda d: dot(d["embedding"], q_vec) /
                      (norm(d["embedding"]) * norm(q_vec)),
        reverse=True,
    )
    return scored[:k]

def best_qa_match(user_q: str, docs: list[dict], min_score: int = 90):
    """
    Return the doc whose meta['Question'] is most similar to user_q
    if the similarity (0-100) is above min_score; otherwise None.
    """
    # Build a list of (question_text, index) for only the QA rows
    choices = [
        (d["meta"]["Question"], i)
        for i, d in enumerate(docs)
        if d["meta"].get("Question")
    ]
    if not choices:
        return None

    # Rapidfuzz finds the best match fast
    match, score, idx = process.extractOne(
        user_q,
        choices,
        scorer=fuzz.token_set_ratio  # good for re-ordering / extra words
    )
    return docs[idx] if score >= min_score else None

def qa_answer(doc: dict) -> str:
    """
    Extract the prepared answer from a canned QA document.
    Assumes doc["content"] looks like
        'Q: ...\\nA: your answer text'
    """
    parts = doc["content"].split("A:", 1)
    return parts[1].strip() if len(parts) > 1 else doc["content"].strip()

# ── SESSION STATE ──────────────────────────────────────────────────────────
st.session_state.setdefault("code",   None)
st.session_state.setdefault("docs",   [])
st.session_state.setdefault("history", [])

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="QuikPick Oracle", layout="centered")
st.title("QuikPick Oracle")

# 1) Select error code
if st.session_state.code is None:
    code = st.text_input("Enter an error code (e.g. 2_4) to begin:")
    if code and ERROR_RE.match(code):
        bundle = docs_for_code(code)
        if bundle:
            st.session_state.code = code
            st.session_state.docs = bundle
            _rerun()
        else:
            st.error("Error code not found.")
    st.stop()

# 2) Banner (fixed backtick removed)
main = next(d for d in st.session_state.docs if "Message" in d["meta"])
with st.expander("Error-code details", expanded=True):
    st.markdown(
        f"**Error Code:** {main['meta']['ErrorCode']}  \n"
        f"**Message:** {main['meta']['Message']}  \n"
        f"**Solution:** {main['meta']['Solution']}"
    )

# 3) Replay chat history AND inject feedback widgets
for i, msg in enumerate(st.session_state.history):
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    else:
        with st.chat_message("assistant", avatar=logo):
            st.markdown(msg.content)

            # ensure meta exists (fallback with no follow‑ups)
            if i not in st.session_state["assistant_meta"]:
                st.session_state["assistant_meta"][i] = {
                    "q": st.session_state.history[i-1].content,
                    "a": msg.content,
                    "fups": "",
                }

            meta      = st.session_state["assistant_meta"][i]
            llm_fups  = meta.get("fups", "")

            # a) LLM‑generated follow‑ups
            if llm_fups:
                st.divider()
                #st.markdown("**LLM Suggested follow‑ups:**")
                lines = [l for l in llm_fups.splitlines() if l.strip()][:3]
                for j, line in enumerate(lines):
                    q = line.lstrip("0123456789.- ").strip()
                    if q:
                        st.button(
                            q,
                            key=f"fup-{j}-{i}",
                            on_click=click_fup,
                            args=(q,),
                        )

            # b) canned Q&A follow‑ups
            #canned = [
            #    d["meta"]["Question"]
            #    for d in st.session_state.docs
            #    if d["meta"].get("IsQA")
            #]
            #if canned:
            #    st.divider()
            #    st.markdown("**Canned follow‑ups:**")
            #    for j, q in enumerate(canned):
            #        st.button(
            #            q,
            #            key=f"qa-{j}-{i}",
            #            on_click=click_fup,
            #            args=(q,),
            #        )            
            
            # ── Unified feedback widget ──
            widget_key   = f"fb_{i}"
            persist_key  = f"fb_score_{i}"
            prev_score   = st.session_state.get(persist_key)
            disable_icon = None if prev_score is None else ("👍" if prev_score == 1 else "👎")

            # Render the component and capture raw response dict
            resp = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Tell us more",
                disable_with_score=disable_icon,
                key=widget_key,
                align="flex-end",
            )

            # On first non‑None resp, convert & save exactly once
            if resp is not None and persist_key not in st.session_state:
                raw_score = resp["score"]
                # 1) try to cast directly (handles "1" or 0/1)
                try:
                    score = int(raw_score)
                except (ValueError, TypeError):
                    # 2) fallback: look for "+1" or "👍" in string
                    s = str(raw_score)
                    score = 1 if ("+1" in s or "👍" in s) else 0

                text   = (resp.get("text") or "").strip()
                rating = 5 if score == 1 else 1

                # Persist to your DB
                save_feedback(
                    st.session_state.code,
                    st.session_state["assistant_meta"][i]["q"],
                    st.session_state["assistant_meta"][i]["a"],
                    rating=rating,
                    comment=text,
                )
                # For 👍 also append the canned QA
                if score == 1:
                    append_qa_if_new(
                        st.session_state.code,
                        st.session_state["assistant_meta"][i]["q"],
                        st.session_state["assistant_meta"][i]["a"],
                    )

                # Remember the numeric score so disable_with_score works correctly next rerun
                st.session_state[persist_key] = score
                st.toast("Thanks for the feedback!")


# 4) New question or follow‑up input (OUTSIDE the for‑loop)

# a) If a follow‑up button was clicked, stage it and lock input
fup = st.session_state.pop("next_q", None)
if fup:
    st.session_state.pending_q   = fup
    st.session_state.is_thinking = True
    _rerun()

# b) Show the input box, disabled if we’re “thinking”
typed = st.chat_input(
    "Ask a question …",
    key="main_input",
    disabled=st.session_state.is_thinking,
)
# If the user typed while not already pending, stage it
if typed and st.session_state.pending_q is None:
    st.session_state.pending_q   = typed
    st.session_state.is_thinking = True
    _rerun()

# c) If there’s a staged prompt, process it
if st.session_state.pending_q:
    user_q = st.session_state.pending_q
    st.session_state.pending_q = None

    # record user message
    st.chat_message("user").write(user_q)
    st.session_state.history.append(HumanMessage(content=user_q))

    # ─ pull context, call LLM ─
    with st.spinner("thinking…"):
        ctx       = retrieve_similar(user_q, st.session_state.docs, k=3)
        ctx_block = "\n\n".join(d["content"] for d in ctx)
        hist_txt  = "\n".join(m.content for m in st.session_state.history[-MEM_TURNS:])
        prompt    = build_prompt(SYSTEM_TEMPLATE, ctx_block, hist_txt, user_q) + " "
        resp      = llm(prompt, max_tokens=MAX_TOKENS,
                        temperature=0.2, top_p=0.95,
                        stop=["<END>"])
        raw       = resp["choices"][0]["text"].strip() # type: ignore
        if "<END>" in raw:
            raw = raw.split("<END>",1)[0].rstrip()
        main_ans, llm_fups = (raw.split("### Follow-Up",1)+[""])[:2]
        main_ans, llm_fups = main_ans.strip(), llm_fups.strip()

    # stash Q/A + follow‑ups for replay
    mid = len(st.session_state.history)
    st.session_state["assistant_meta"][mid] = {
        "q":     user_q,
        "a":     main_ans,
        "fups":  llm_fups,
    }

    # append AI reply
    st.session_state.history.append(AIMessage(content=main_ans))

    # unlock input & rerun to redraw the enabled chat box
    st.session_state.is_thinking = False
    _rerun()

for kind, payload in st.session_state.pop("to_log", []):
    if kind == "pos":
        log_positive(*payload)   # type: ignore[arg-type]
    else:
        log_negative(*payload)   # type: ignore[arg-type]

# 5) Reset
if st.button("🔄 Start over"):
    st.session_state.clear()
    _rerun()