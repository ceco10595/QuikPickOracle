"""
Ingest • error_codes.csv  (required)
      • sample_qa.csv     (optional)

Usage:
    python src/ingest.py  data/error_codes.csv  [data/sample_qa.csv]
"""
import sys, pathlib
import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR  = "vectorstore"
COLL_NAME   = "errors"

def embed(texts):
    model = SentenceTransformer(EMBED_MODEL, device="mps")
    return model.encode(texts, batch_size=64, show_progress_bar=True)

def normalise_codes(df: pd.DataFrame) -> pd.DataFrame:
    rename = {"Error Code":"ErrorCode", "Error Message":"Message"}
    return df.rename(columns=rename).fillna("")

def ingest(code_csv: pathlib.Path, qa_csv: pathlib.Path | None = None):
    # ── load & format ────────────────────────────────────────────────────────
    code_df = normalise_codes(pd.read_csv(code_csv))
    code_docs = code_df.apply(
        lambda r: f"Error Code: {r['ErrorCode']}\nMessage: {r['Message']}\n"
                  f"Solution: {r['Solution']}", axis=1).tolist()

    if qa_csv and qa_csv.exists():
        qa_df = pd.read_csv(qa_csv)[["ErrorCode","Question","Answer"]].fillna("")
        qa_docs = qa_df.apply(lambda r: f"Q: {r['Question']}\nA: {r['Answer']}",
                              axis=1).tolist()
    else:
        qa_df, qa_docs = pd.DataFrame(), []

    # ── embed ───────────────────────────────────────────────────────────────
    vec_code = embed(code_docs)
    vec_qa   = embed(qa_docs) if qa_docs else []

    # ── store ───────────────────────────────────────────────────────────────
    client = PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(COLL_NAME)

    col.add(ids=[f"code-{i}" for i in range(len(code_docs))],
            documents=code_docs, embeddings=vec_code,
            metadatas=code_df.to_dict("records"))

    if qa_docs:
        col.add(ids=[f"qa-{i}" for i in range(len(qa_docs))],
                documents=qa_docs, embeddings=vec_qa,
                metadatas=qa_df.assign(IsQA=True).to_dict("records"))

    print(f"✅ Indexed {len(code_docs)} codes + {len(qa_docs)} QAs")

if __name__ == "__main__":
    code_csv = pathlib.Path(sys.argv[1])
    qa_csv   = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    ingest(code_csv, qa_csv)
