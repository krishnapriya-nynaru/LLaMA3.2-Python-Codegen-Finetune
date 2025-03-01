import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Load original (unfine-tuned) model
base_model_id_original = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def format_sample(sample):
    """ Helper function to format a single input sample"""
    instruction=sample['instruction']
    input_text=sample['input']

    if input_text is None or input_text=="":
        formatted_prompt=(
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        formatted_prompt=(
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n"
            f"### Response:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    formatted_prompt="".join(formatted_prompt) # exclude trailing white spaces
    return formatted_prompt                    # stream text into the dataloader, one by one


def generate_ft(model, sample, tokenizer, max_new_tokens, context_size=256, temperature=0.0, top_k=1, eos_id=[128001, 128009]):
    """
    Generate text using a language model with proper dtype handling
    """
    # Get model's expected dtype and device
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    formatted_prompt = format_sample(sample)

    # Encode and prepare input
    idx = tokenizer.encode(formatted_prompt)
    idx = torch.tensor(idx, dtype=torch.long, device=model_device).unsqueeze(0)
    num_tokens = idx.shape[1]

    # Generation loop
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   # conditioning context
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids=idx_cond,use_cache=False)
            logits = outputs.logits

        # Focus on last time step
        logits = logits[:, -1, :]

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, [-1]]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=model_device, dtype=model_dtype),
                logits
            )

        # Apply temperature and sample
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Check for EOS
        if idx_next.item() in eos_id:
            break

        # Append new token
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode generated text
    generated_ids = idx.squeeze(0)[num_tokens:]
    generated_text = tokenizer.decode(generated_ids)

    return generated_text
def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

# Load test dataset
def load_test_data(filename="test_data.json"):
    with open(filename, "r") as file:
        return json.load(file)

# Evaluate responses using original model
def evaluate_responses(test_data, model, tokenizer, filename="evaluation_results.json"):
    for sample in tqdm(test_data, desc="Evaluating responses"):
        sample["original model response"] = generate_ft(model, sample, tokenizer, max_new_tokens=256)
    
    with open(filename, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Evaluation results saved to {filename}")

if __name__ == "__main__":
    print("Loading original LLaMA-3.2-1B model...")
    model_original, tokenizer_original = load_model(base_model_id_original)
    
    print("Loading test dataset...")
    test_samples = load_test_data()
    
    print("Evaluating responses...")
    evaluate_responses(test_samples, model_original, tokenizer_original)
