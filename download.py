#!/usr/bin/env python3
"""
download_distilgpt2.py

Fetches Hugging Face’s distilgpt2 model and tokenizer,
adds a pad token if missing, and saves both locally
for inference-only use.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME   = "distilgpt2"
OUTPUT_DIR   = "./distilgpt2-finetuned-dadjokes"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load tokenizer, add pad token if absent
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        print("Added [PAD] token to tokenizer.")

    # 2) Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 3) Resize embeddings (in case pad token was added)
    model.resize_token_embeddings(len(tokenizer))

    # 4) Save both to disk
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print(f"✅ Model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()