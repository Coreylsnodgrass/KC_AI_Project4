import json
from pathlib import Path

# Define file paths
INPUT = Path("data.txt")
OUTPUT = Path("data.jsonl")

# Set to track unique lines
seen = set()

# Track total token count
token_count = 0
MAX_TOKENS = 100_000  # Token limit for the entire dataset

# Open input and output files
with INPUT.open("r", encoding="utf-8", errors="ignore") as fin, \
     OUTPUT.open("w", encoding="utf-8") as fout:

    for line in fin:
        text = line.strip().lower()

        # Skip empty or duplicate lines
        if not text or text in seen:
            continue

        tokens = text.split()

        # Stop if token cap would be exceeded
        if token_count + len(tokens) > MAX_TOKENS:
            break

        seen.add(text)
        token_count += len(tokens)

        # Write cleaned and tokenized line to output
        clean = " ".join(tokens)
        fout.write(json.dumps({"text": clean}, ensure_ascii=False) + "\n")

# Summary log
print(f"Written {len(seen)} unique lines ({token_count} tokens) to {OUTPUT}")
