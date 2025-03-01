# 🚀 Model Download & Setup Guide  

This directory contains model-related files. However, due to size constraints, the **LLaMA 3.2-1B (2.45GB) and the fine-tuned model are not included in this repository.**  
Follow the steps below to download and set them up for training, evaluation, and inference.  

---

### 🔹 Download Base Model (Original LLaMA 3.2-1B)  

The original **LLaMA 3.2-1B-Instruct** model can be downloaded from **Hugging Face**:  

```bash
huggingface-cli login  # Authenticate your Hugging Face account  
git lfs install  
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct models/base_model
```

Alternatively, you can download it manually:
[***Meta LLaMA 3.2-1B-Instruct***](https://www.llama.com/)


# 🚀 Download Fine-Tuned LLaMA 3.2-1B Model  

The fine-tuned **LLaMA 3.2-1B model** was trained on the **`iamtarun/python_code_instructions_18k_alpaca`** dataset using **QLoRA** and **SFTTrainer**.  

---

### 🔹 Download Fine-Tuned Model  

To download the fine-tuned model from Hugging Face, use the following commands:  

```bash
huggingface-cli login  
git clone https://huggingface.co/priyanynaru/LLaMA3.2-Python-Codegen-Finetune
```

### 🔹Manual Download
Alternatively, you can download the model manually from Hugging Face:
[🔗 Fine-Tuned LLaMA 3.2-1B Model](https://huggingface.co/priyanynaru/LLaMA3.2-Python-Codegen-Finetune)

### 🔹Load the Models in Your Code
Modify your scripts to load the downloaded models:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer  

base_model_path = "./models/base_model"  
finetuned_model_path = "./models/finetuned_model"  

# Load Base Model  
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)  
tokenizer = AutoTokenizer.from_pretrained(base_model_path)  

# Load Fine-Tuned Model  
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)  
tokenizer_ft = AutoTokenizer.from_pretrained(finetuned_model_path)  
```

### 📂 Expected Directory Structure
Once downloaded, your models/ directory should be structured as follows:
```bash
models/  
│── base_model/             # Original LLaMA 3.2-1B model  
│── finetuned_model/        # Fine-tuned model checkpoint  
│── README.md               # This file with download instructions 
```