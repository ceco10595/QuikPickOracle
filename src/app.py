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
from rapidfuzz import fuzz, process   # pip install rapidfuzz
st.session_state.setdefault("to_log", [])   # list of ('pos'|'neg', payload)
logo = Image.open("src/new-Pepsi-logo-png.png")

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
# helpers.py  (or drop it near the top of app.py)
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
    # 1️⃣  fuzzy QA match first
    qa_hit = best_qa_match(q, docs, min_score=90)
    if qa_hit:
        return [qa_hit]                       # give the “gold” answer only

    # 2️⃣  otherwise fall back to embedding similarity
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

# 3) Show chat history (with PepsiCo logo avatar for assistant)
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        # user messages get the default user icon
        st.chat_message("user").markdown(msg.content)
    else:
        # assistant messages now carry your logo
        st.chat_message("assistant", avatar=logo).markdown(msg.content)


# follow-up helper
def click_fup(q: str):
    st.session_state.next_q = q

# 4) New question or follow-up button
# a) pull any queued follow-up
follow = st.session_state.pop("next_q", None)

# b) always show the chat_input widget
typed  = st.chat_input("Ask a question …")

# C) pick the follow-up if present, otherwise what the user typed
user_q = follow or typed


if user_q:
    # record question
    st.chat_message("user").write(user_q)
    st.session_state.history.append(HumanMessage(content=user_q))

    # retrieve context
    ctx = retrieve_similar(user_q, st.session_state.docs, k=3)
    ctx_block = "\n\n".join(d["content"] for d in ctx)
    ctx_docs = retrieve_similar(user_q, st.session_state.docs, k=3)

    # build prompt (history w/o role-tokens)
    if len(ctx_docs) == 1 and ctx_docs[0]["meta"].get("IsQA"):
        main_ans = qa_answer(ctx_docs[0])      # <- pulls the “A: …” line
        llm_fups = ""                          # no follow-ups for canned reply
    else:
        ctx_block = "\n\n".join(d["content"] for d in ctx)
        hist_txt = "\n".join(m.content for m in st.session_state.history[-MEM_TURNS:])
        prompt = build_prompt(SYSTEM_TEMPLATE, ctx_block, hist_txt, user_q) + " "

        # call LLM
        with st.spinner("thinking…"):
            try:
                resp = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.2, top_p=0.95)
            except Exception as e:
                st.exception(e)
                st.stop()

            
    raw = resp["choices"][0]["text"].strip() # type: ignore

    # ─hard-stop at the sentinel
    if "<END>" in raw:
        raw = raw.split("<END>", 1)[0].rstrip()

    if not raw:
        st.error("⚠️"); st.stop()

    # split out LLM’s follow-ups
    main_ans, llm_fups = (raw.split("### Follow-Up", 1) + [""])[:2]
    main_ans, llm_fups = main_ans.strip(), llm_fups.strip()

    # assistant bubble
    with st.chat_message("assistant", avatar = logo): #avatar = logo
        st.markdown(main_ans)

        # a) LLM‐generated
        if llm_fups:
            st.divider()
            st.markdown("**LLM Suggested follow-ups:**")
            # only take the first 3 non‐empty lines
            lines = [l for l in llm_fups.splitlines() if l.strip()][:3]
            for i, line in enumerate(lines):
                q = line.lstrip("0123456789.- ").strip()
                if q:
                    st.button(q, key=f"llm-{i}-{len(st.session_state.history)}",
                              on_click=click_fup, args=(q,))

        # b) canned Q&A
        canned = [d["meta"]["Question"] for d in st.session_state.docs if d["meta"].get("IsQA")]
        if canned:
            st.divider()
            st.markdown("**Canned follow-ups:**")
            for i, q in enumerate(canned):
                st.button(q, key=f"qa-{i}-{len(st.session_state.history)}",
                          on_click=click_fup, args=(q,))

        #c) Feedback Loop
        def _thumbs_up(code, q, a, mid):
            save_feedback(code, q, a, rating=5)
            append_qa_if_new(code, q, a)
            st.session_state[f"fb_done_{mid}"] = True

        def _thumbs_down(code, q, a, mid):
            st.session_state[f"show_cmt_{mid}"] = True

        mid = len(st.session_state.history)

        col_up, col_down = st.columns(2)
        col_up.button("👍", key=f"up_{mid}",
                    on_click=_thumbs_up,
                    args=(st.session_state.code, user_q, main_ans, mid))

        col_down.button("👎", key=f"down_{mid}",
                        on_click=_thumbs_down,
                        args=(st.session_state.code, user_q, main_ans, mid))

        if st.session_state.get(f"show_cmt_{mid}", False):
            txt = st.text_input("Describe the issue", key=f"cmt_{mid}")
            if st.button("Submit", key=f"sub_{mid}") and txt.strip():
                save_feedback(st.session_state.code, user_q, main_ans,
                            rating=1, comment=txt.strip())
                st.success("Thanks – feedback recorded!")



    # save reply
    st.session_state.history.append(AIMessage(content=main_ans))

for kind, payload in st.session_state.pop("to_log", []):
    if kind == "pos":
        log_positive(*payload)   # type: ignore[arg-type]
    else:
        log_negative(*payload)   # type: ignore[arg-type]

# 5) Reset
if st.button("🔄 Start over"):
    st.session_state.clear()
    _rerun()