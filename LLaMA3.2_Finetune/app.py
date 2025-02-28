import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaGenerator:
    def __init__(self, model_path, device_map=None):
        self.model_path = model_path
        self.device_map = device_map if device_map else {'': 0}
        self.model, self.tokenizer = self.load_model()
    
    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return model, tokenizer
    
    def generate_text(self, prompt, max_new_tokens=256, context_size=1024, temperature=0.0, top_k=1, eos_id=[128001, 128009]):
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

# Streamlit UI
st.title("LLaMA 3.2 Python Code Generator")
st.write("Generate Python code snippets instantly using LLaMA 3.2. Just enter a prompt describing the functionality you need!")

model_path = "./llama-3.2-1b-custom/model/"
generator = LlamaGenerator(model_path)

st.sidebar.header("Generation Parameters")
max_tokens = st.sidebar.slider("Max new tokens", min_value=32, max_value=512, value=128)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
top_k = st.sidebar.slider("Top-K sampling", min_value=1, max_value=50, value=10)

prompt = st.text_area("Enter your prompt:")

if st.button("Generate Code"):
    if prompt:
        with st.spinner("Generating response..."):
            output = generator.generate_text(prompt, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        st.subheader("Generated Response:")
        st.code(output, language='python')
    else:
        st.warning("Please enter a prompt before generating text.")
