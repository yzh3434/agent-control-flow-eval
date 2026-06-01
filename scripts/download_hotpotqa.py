"""Download a difficulty-labeled HotpotQA pool (distractor) and convert to our JSON.

HotpotQA's dev/test sets are *all* labeled "hard" by design; the easy/medium/hard
`level` labels only exist in the train split. Since we never train a model (we only
evaluate control flows), we source a difficulty-labeled evaluation pool from the
train distractor split, capping each level so the file stays small.

Output row shape (matches the env):
  {"_id", "question", "answer", "level", "type", "context": [[title, [sent...]], ...]}
"""
import io
import json
import os
import sys

import requests
import pyarrow as pa
import pyarrow.parquet as pq

URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
       "refs%2Fconvert%2Fparquet/distractor/train/0000.parquet")
OUT = os.path.join("data", "hotpot_distractor_labeled.json")
LEVELS = ("easy", "medium", "hard")
CAP_PER_LEVEL = 500


def _select_indices(levels):
    counts = {lv: 0 for lv in LEVELS}
    indices = []
    for i, lv in enumerate(levels):
        if lv in counts and counts[lv] < CAP_PER_LEVEL:
            counts[lv] += 1
            indices.append(i)
    return indices


def _to_rows(table):
    rows = []
    for r in table.to_pylist():
        ctx = r["context"]
        context = [[title, sents]
                   for title, sents in zip(ctx["title"], ctx["sentences"])]
        rows.append({
            "_id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "level": r["level"],
            "type": r["type"],
            "context": context,
        })
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        print(f"Already present: {OUT}")
        return 0
    print("Downloading HotpotQA (distractor, train) parquet from HuggingFace ...")
    try:
        content = requests.get(URL, timeout=600).content
    except requests.RequestException as err:
        print(f"\nDownload failed: {err}")
        print("If HuggingFace is blocked, set HF_ENDPOINT to a mirror (e.g. "
              "https://hf-mirror.com) or download the parquet manually, then re-run.")
        return 1
    table = pq.read_table(io.BytesIO(content))
    indices = _select_indices(table.column("level").to_pylist())
    subset = table.take(pa.array(indices))
    rows = _to_rows(subset)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    levels = {}
    for r in rows:
        levels[r["level"]] = levels.get(r["level"], 0) + 1
    print(f"Saved {len(rows)} questions to {OUT}")
    print(f"Difficulty distribution: {levels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
