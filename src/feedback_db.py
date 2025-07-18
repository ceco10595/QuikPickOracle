# src/feedback_db.py
from pathlib import Path
import sqlite3, time, threading

import csv, json
_EXTRA_QA = Path("data/sample_qa.csv")
_EXTRA_NEG = Path("fine-tune/negatives.jsonl")

DB_PATH = Path("feedback.db")
_LOCK   = threading.Lock()

def _init():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL,
                error_code TEXT,
                question   TEXT,
                answer     TEXT,
                rating     INTEGER,   -- 5 = thumbs-up, 1 = thumbs-down
                comment    TEXT
            )
        """)

def save(code: str, question: str, answer: str,
         *, rating: int, comment: str | None = None) -> None:
    """Thread-safe insert."""
    _init()
    print("CALLED save():", code, rating)   #  <-- TEMP debug
    with _LOCK, sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO feedback VALUES (NULL,?,?,?,?,?,?)",
            (time.time(), code, question, answer, rating, comment)
        )
# ---------- OPTIONAL EXTRA LOGS ----------


def _append_positive(code: str, q: str, a: str):
    _EXTRA_QA.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, _EXTRA_QA.open("a", newline="") as f:
        csv.writer(f).writerow([code, q.strip(), a.strip()])

def _append_negative(code: str, q: str, a: str, fb: str):
    _EXTRA_NEG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "error_code": code,
        "instruction": q.strip(),
        "response": a.strip(),
        "feedback": fb.strip() or "<no comment>",
    }
    print("Appending negative feedback:", payload)  # <--- Add this line
    with _LOCK, _EXTRA_NEG.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")