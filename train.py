# train.py
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_NAME   = "distilgpt2"
OUTPUT_DIR   = "./distilgpt2-finetuned-dadjokes"
TRAIN_FILE   = "data.jsonl"      # from prepare_data.py
NUM_EPOCHS   = 2
MAX_LENGTH   = 128
BATCH_SIZE   = 8

def main():
    # 1) load your JSONL of jokes
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

    # 2) tokenizer (ensure there’s a pad token)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 3) tokenize
    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)
    tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 4) split off 10% for eval
    split = tok_ds.train_test_split(test_size=0.1)

    # 5) model + data collator
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 6) training args (old-style evaluation)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        do_eval=True,        # enable evaluation
        eval_steps=500,      # how often to run eval
        logging_steps=50,    # how often to log
        save_steps=500,      # how often to checkpoint
        save_total_limit=2,  # keep only the last 2 checkpoints
    )

    # 7) trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8) fine-tune
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
