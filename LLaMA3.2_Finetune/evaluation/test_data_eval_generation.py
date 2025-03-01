import json
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
base_model_id = "./llama-3.2-1b-custom/model/"
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
    formatted_prompt="".join(formatted_prompt) 
    return formatted_prompt                    


def generate_ft(model, sample, tokenizer, max_new_tokens, context_size=256, temperature=0.0, top_k=1, eos_id=[128001, 128009]):
    """
    Generate text using a language model with proper dtype handling
    """
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    formatted_prompt = format_sample(sample)

    # Encode and prepare input
    idx = tokenizer.encode(formatted_prompt)
    idx = torch.tensor(idx, dtype=torch.long, device=model_device).unsqueeze(0)
    num_tokens = idx.shape[1]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   
        with torch.no_grad():
            outputs = model(input_ids=idx_cond,use_cache=False)
            logits = outputs.logits

        logits = logits[:, -1, :]

        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, [-1]]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=model_device, dtype=model_dtype),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next.item() in eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    # Decode generated text
    generated_ids = idx.squeeze(0)[num_tokens:]
    generated_text = tokenizer.decode(generated_ids)

    return generated_text

def load_model():
    model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    return model, tokenizer

# Load dataset and select 100 random samples
def load_dataset_samples():
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    return random.sample(list(ds), 100)  

# Generate response using fine-tuned model
def generate_response(model, tokenizer, sample):
    return generate_ft(model, sample, tokenizer, max_new_tokens=256)

# Save test dataset with model responses
def save_test_data(samples, model, tokenizer, filename="test_data.json"):
    for sample in samples:
        sample["model response"] = generate_response(model, tokenizer, sample)
    
    with open(filename, "w") as file:
        json.dump(samples, file, indent=4)
    print(f"Test data saved to {filename}")

if __name__ == "__main__":
    print("Loading fine-tuned model...")
    model, tokenizer = load_model()
    
    print("Loading dataset samples...")
    test_samples = load_dataset_samples()
    
    print("Generating model responses...")
    save_test_data(test_samples, model, tokenizer)
