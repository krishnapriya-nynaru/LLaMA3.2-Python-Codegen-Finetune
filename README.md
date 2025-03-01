# LlaMA3.2-Python-Codegen-Finetune
LLaMA3.2-Python-Codegen-Finetune provides a complete pipeline for fine-tuning, evaluating, and deploying a LLaMA 3.2 model for Python code generation. This repository includes fine-tuning scripts, model evaluation, testing with real-world Python code tasks, a Streamlit-based UI, and a comparison of fine-tuned vs. original LLaMA 3.2 1B model.

## Table of Contents
- [Key Features](#key-features)
- [Dataset Used](#dataset-used)
- [Installation](#installation)
- [Training Details](#training-details)
- [Testing & Evaluation](#testing)
- [Repository Structure](#repository)
- [Results & Model Performance](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Key Features:
- **📝 Fine-Tuning LLaMA 3.2 on Python Code -** Uses QLoRA + SFTTrainer for memory-efficient fine-tuning.
- **📊 Model Evaluation -** Compares Fine-Tuned vs. Original LLaMA using BLEU & ROUGE scores.
- **🧪 Automated Testing -** Evaluates Python code generation using 100 real-world programming tasks.
- **🌐 Streamlit UI -** Interactive UI for testing model responses in real time.
- **📈 Benchmarking Against Base Model -** Ensures performance improvements over the original LLaMA 3.2 model.
- **🤝 Hugging Face Integration -** Supports pushing fine-tuned models to Hugging Face Model Hub.

## Installation
1. Create conda enviroenment 
    ```bash
    conda create -n env_name python=3.10
2. Activate conda enviroenment
    ```bash
    conda activate env_name
3. Clone this repository:
   ```bash
   git clone https://github.com/krishnapriya-nynaru/LLaMA3.2-Python-Codegen-Finetune.git
4. Change to Project directory
    ```bash
    cd LLaMA3.2_Finetune
5. Install required packages :
    ```bash
    pip install -r requirements.txt
6. Login to Hugging Face
    ```bash
    huggingface-cli login
    ```

***Note*** : You can create Hugging face ***Token*** from this [link](https://huggingface.co/settings/tokens).

## Training Details
Run script to start the model training.
```bash
python train.py
```
- ✔️QLoRA (Quantized LoRA) is used to train efficiently with low VRAM usage.
- ✔️ SFTTrainer simplifies training with PEFT & QLoRA integration.

#### Training Logging via Weights & Biases (W&B)
```bash
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: Logging into wandb.ai...
```
***Note*** : Get your w&b ***API key*** this [link](https://wandb.ai/authorize)

#### 📂 Configuration File:

The full training configuration can be found in **config/config.json.**

This file contains model settings, dataset paths, training hyperparameters, and optimization techniques.

## Testing & Evaluation
- Step 1: Test Fine-tuned model 
    ```bash
    python test/test.py
    ```
***Note -***  You can test this with my fine-tuned model by running:

    python test/test_custom_model

- Step 2: Generate Test Data 
    ```bash
    python test_data_eval_generation
    ```
- step 3: Evaluate Responses Using Original LLaMA 3.2 1B Model
    ```bash
    python evaluation/evaluation_data.py
    ```
- Step 4: Compare Fine-Tuned Model vs. Original Model
    ```bash
    python evaluation/evaluate.py
    ```
## Repository Structure 
```bash
LLaMA3.2-Python-Codegen-Finetune/
│── config/                    # Configuration files  
│   ├── config.json            # Training hyperparameters  
│── evaluation/                # Model evaluation scripts & data  
│   ├── eval_data/             # Stores evaluation results  
│   ├── evaluate.py            # Compares fine-tuned vs original model  
│   ├── evaluation_data.py     # Generates responses from original model  
│   ├── test_data_eval_generation.py  # Creates test dataset  
│── models/                    # Model checkpoints & download instructions  
│   ├── README.md              # Guide to download models  
│── testing/                   # Scripts for model testing  
│   ├── test.py                # Tests base model performance  
│   ├── test_custom_model.py   # Tests fine-tuned model  
│── app.py                     # Streamlit UI for testing  
│── requirements.txt           # Package dependencies  
│── train.py                   # Fine-tuning script  
 
```

## Results and Model performance

### Input to Finetuned Model
Given the participants’ score sheet for your University Sports Day, you must find the runner-up score. You are given n scores. Store them in a list and find the score of the runner-up.

***Input Format***

The first line contains n. The second line contains an array A[] of n integers are each separated by a space.

Constraints

2 ≤  n ≤  10

-100 ≤  A[i] ≤ 100

***Output Format***

Print the runner-up score.

Sample Input 

5 2 3 6 6 5

Sample Output 

5

Explanation 

The given list is [2, 3, 6, 6, 5]. The maximum score is 6, the second maximum score is 5. Hence, we print 5 as the runner-up score.

### Model response
```python
n = int(input())
scores = list(map(int, input().split(" ")))
scores.sort()
print(scores[1])
#### Output: 5
#### Explanation: The given list is [2, 3, 6, 6, 5]. The maximum score is 6, the second maximum score is 5. Hence, we print 5 as the runner-up score.

#### This is because the 5th element in the sorted list, which is 5, is the second highest score. Therefore, we print 5 as the runner-up score.

#### The output of the program is 5

####The time complexity of the program is O(nlogn) as the array sorts on the fly. The space complexity is O(1) as no extra data structures are required to perform the sorting.
```
![alt_text](https://github.com/krishnapriya-nynaru/LLaMA3.2-Python-Codegen-Finetune/blob/main/LLaMA3.2_Finetune/results/response.png?raw=true)

### 🏆 Evaluation: Fine-Tuned vs. Original LLaMA-3.2-1B  
#### 🔍 Initial Testing Results  

| Model                      | BLEU Score | ROUGE Score |
|----------------------------|------------|-------------|
| Fine-Tuned LLaMA-3.2-1B    | **3.32**   | **24.16**   |
| Original LLaMA-3.2-1B      | 2.51       | 17.92       |

![alt_text](https://github.com/krishnapriya-nynaru/LLaMA3.2-Python-Codegen-Finetune/blob/main/LLaMA3.2_Finetune/results/evaluation.png?raw=true)
#### 🔬 After Further Evaluation  

| Model                      | BLEU Score | ROUGE Score |
|----------------------------|------------|-------------|
| Fine-Tuned LLaMA-3.2-1B    | **4.30**   | **27.42**   |
| Original LLaMA-3.2-1B      | 3.71       | 22.10       |
![alt_text](https://github.com/krishnapriya-nynaru/LLaMA3.2-Python-Codegen-Finetune/blob/main/LLaMA3.2_Finetune/results/eval.png?raw=true)

✅ **QLoRA + SFTTrainer improved BLEU & ROUGE scores by 15-20% on Python coding tasks!** 🚀  

## Contributing 
Contributions are welcome! To contribute to this project:
1. Fork the repository.
2. Create a new branch for your changes.
3. Commit changes & push to GitHub.
4. Submit a pull request with a detailed description of your changes.

If you have any suggestions for improvements or features, feel free to open an issue!

## Acknowledgments  

💡 Powered by:

- [**LLaMA3.2**](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) 
- [**QLoRA**](https://github.com/huggingface/peft) 
- [**Weights & Bias**](https://wandb.ai/site/)
- [**SFTTrainer**](https://huggingface.co/docs/trl/en/sft_trainer)
- [**Hugging Face Transformers**](https://huggingface.co/docs/transformers/en/index)
- [**Streamlit**](https://streamlit.io/)

🚀 Fine-tune, test, and deploy your AI-powered Python Code Generator today!





