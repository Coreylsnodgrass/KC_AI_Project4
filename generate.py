from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer from the specified directory
MODEL_DIR = "./distilgpt2-finetuned-dadjokes"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)


def generate_text(prompt: str,
                  max_length: int = 80,
                  temperature: float = 0.7,
                  top_k: int = 50,
                  top_p: float = 0.9) -> str:
    """
    Generate text continuation for a given prompt using the fine-tuned model.

    Args:
        prompt (str): The input text prompt.
        max_length (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature; higher values increase diversity.
        top_k (int): Top-K sampling; limits the next-token candidates to the k most probable.
        top_p (float): Top-p (nucleus) sampling; cumulative probability threshold.

    Returns:
        str: The generated text, cropped at the first period.
    """
    # Encode the prompt into token IDs and attention mask tensors
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Perform sampling-based generation
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )

    # Remove prompt tokens from the generated output
    generated_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Crop the output at the first period to form a complete sentence
    if "." in text:
        text = text.split(".", 1)[0] + "."

    return text
