import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Constants for configuration
MODEL_NAME   = "distilgpt2"                         # Lightweight pretrained GPT-2 model
OUTPUT_DIR   = "./distilgpt2-finetuned-dadjokes"    # Where to save the fine-tuned model
TRAIN_FILE   = "data.jsonl"                         # Input dataset (from prepare_data.py)
NUM_EPOCHS   = 2                                     # Keep small for demo
MAX_LENGTH   = 128                                   # Max token length per sequence
BATCH_SIZE   = 8                                     # Batch size per device

def main():
    # Step 1: Load dataset from JSONL
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE})["train"]

    # Step 2: Load tokenizer and ensure [PAD] token exists
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Step 3: Tokenize the dataset
    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)
    tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Step 4: Split dataset into 90% train, 10% eval
    split = tok_ds.train_test_split(test_size=0.1)

    # Step 5: Load model and attach data collator for language modeling
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))  # Resize in case PAD token was added
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Step 6: Define training parameters and checkpoints
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        do_eval=True,          # Enables evaluation
        eval_steps=500,        # Run evaluation every 500 steps
        logging_steps=50,      # Log every 50 steps
        save_steps=500,        # Save model every 500 steps
        save_total_limit=2,    # Keep only last 2 checkpoints
    )

    # Step 7: Set up Trainer with training and evaluation datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Step 8: Train and save the model and tokenizer
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
