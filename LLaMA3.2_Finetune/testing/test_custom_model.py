import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaGenerator:
    def __init__(self, model_name, device_map=None):
        self.model_name = model_name
        self.device_map = device_map if device_map else {'': 0}
        self.model, self.tokenizer = self.load_model()
    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
    
    def generate_text(self, prompt, max_new_tokens=256, context_size=512, temperature=0.0, top_k=1, eos_id=[128001, 128009]):
        """
        Generate text using the fine-tuned model with improved sampling and handling.
        """
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device

        formatted_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n"
            f"### Response:\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        idx = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(model_device)
        num_tokens = idx.shape[1]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            
            with torch.no_grad():
                outputs = self.model(input_ids=idx_cond, use_cache=False)
                logits = outputs.logits[:, -1, :]
            
            if top_k and top_k > 0:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, [-1]]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf'), device=model_device, dtype=model_dtype), logits)
            
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            if idx_next.item() in eos_id:
                break
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        generated_text = self.tokenizer.decode(idx.squeeze(0)[num_tokens:])
        return generated_text

if __name__ == "__main__":
    model_name = "priyanynaru/LLaMA3.2-Python-Codegen-Finetune"
    generator = LlamaGenerator(model_name)
    prompt1 = """Write a python code to print fibbinoci sequence"""
    output = generator.generate_text(prompt1, max_new_tokens=128)
    print(output)
    print("------------------------------------------------------------------------------------------")
    prompt2 = """Write a python code to add two numbers without using 3rd variable"""
    output = generator.generate_text(prompt2, max_new_tokens=64)
    print(output)