from transformers import AutoTokenizer, AutoModelForCausalLM

# Point this at wherever your fine-tuned model lands
MODEL_DIR = "./distilgpt2-finetuned-dadjokes"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

def generate_text(
    prompt: str,
    max_length: int = 80,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    # strip off the prompt tokens
    generated = out_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── CROP AT FIRST PERIOD ──
    if "." in text:
        text = text.split(".", 1)[0] + "."

    return text
