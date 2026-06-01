"""Download the official GSM8K test split to data/gsm8k_test.jsonl."""
import os
import sys

import requests

URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
       "master/grade_school_math/data/test.jsonl")
OUT = os.path.join("data", "gsm8k_test.jsonl")


def main():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        print(f"Already present: {OUT}")
        return
    print(f"Downloading GSM8K test split from {URL} ...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(resp.text)
    n = sum(1 for line in resp.text.splitlines() if line.strip())
    print(f"Saved {n} questions to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
