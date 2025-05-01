# KC_AI_Project4

# README – Jesterbot Fine-Tuning Project

## a. Model Architecture

This project uses **DistilGPT-2**, a lightweight and distilled version of OpenAI's GPT-2 language model. It is a transformer-based causal language model pretrained on a large corpus of English text. We selected DistilGPT-2 for:

- Efficient performance on CPU-only systems (under 1GB RAM)
- Compatibility with Hugging Face’s Transformers library
- Proven ability to generate coherent sequences

We did **not** define a custom model architecture from scratch. Instead, we fine-tuned this pretrained transformer to specialize in joke generation using our curated dataset.

---

## b. Training Procedure and Loss Behavior

Training was performed using Hugging Face’s `Trainer` API with the following settings:

- **Model**: `distilgpt2`
- **Optimizer**: AdamW (provided internally by `Trainer`)
- **Epochs**: 2
- **Batch size**: 8
- **Token length**: 128 tokens max per input
- **Evaluation**: 10% of training data used as validation
- **Loss function**: CrossEntropyLoss

The training loop used Hugging Face’s logging, evaluation, and checkpointing mechanisms. Loss steadily decreased across epochs, suggesting that the model was learning the patterns present in the dad joke data.

---

## c. How the Tokenizer Works

We used the **DistilGPT-2 tokenizer**, which is a byte pair encoding (BPE) tokenizer. Key characteristics:

- Handles both tokenization and decoding
- Truncates sequences to a max length of 128 tokens
- `[PAD]` token was manually added during setup (via `download.py`)
- Saved locally and reused during inference to maintain consistency

Tokenization was applied using `tokenizer(ex["text"], truncation=True, max_length=128)`.

---

## d. Results and Limitations

### ✅ Results

- The model generates joke-like text when given prompts like:
  - `"Why did the chicken cross the road?"`
  - `"Tell me a joke about dogs."`
- Works well in an interactive CLI setting (`interface.py`)
- Understands some basic setups and delivers punchline-style continuations

### ⚠️ Limitations

- Some outputs repeat words or produce incoherent text due to the dataset’s informal and short nature
- Short training time (2 epochs) limits generalization
- Output is highly sensitive to prompt phrasing
- Not always “funny” or grammatically complete

---

## e. Deployment Steps

### 1. Prepare the data
Run:

```bash
python prepare_data.py
```

### 2. Download base model
Run:

```bash
python download.py
```

### 3.  FIne tune model
Run:

```bash
python train.py
```

### 3.  Run CLI
Run:

```bash
python interface.py
```

