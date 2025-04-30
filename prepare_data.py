import json
from pathlib import Path

# Paths
INPUT = Path("data.txt")
OUTPUT = Path("data.jsonl")

seen = set()
token_count = 0
MAX_TOKENS = 100_000

with INPUT.open("r", encoding="utf-8", errors="ignore") as fin, \
     OUTPUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        text = line.strip().lower()
        if not text or text in seen:
            continue
        tokens = text.split()
        # Stop if adding this line would exceed the token cap
        if token_count + len(tokens) > MAX_TOKENS:
            break
        seen.add(text)
        token_count += len(tokens)
        clean = " ".join(tokens)
        fout.write(json.dumps({"text": clean}, ensure_ascii=False) + "\n")

print(f"Written {len(seen)} unique lines ({token_count} tokens) to {OUTPUT}")
