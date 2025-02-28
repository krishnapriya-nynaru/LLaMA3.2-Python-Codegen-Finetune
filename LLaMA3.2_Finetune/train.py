import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from datasets import load_dataset, Dataset

class LlamaTrainer:
    def __init__(self, config_path="config/config.json"):
        self.load_config(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = self.load_model()
        self.dataset_train, self.dataset_val = self.load_datasets()
        self.training_args = self.get_training_arguments()
        self.trainer = self.get_trainer()
    
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config["use_4bit"],
            bnb_4bit_quant_type=self.config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=self.config["use_nested_quant"],
        )
        return AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            quantization_config=bnb_config,
            device_map="auto"
        )
    
    def format_sample(self, sample):
        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]
        
        formatted_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output_text}\n<|eot_id|>"
        )
        return formatted_prompt
    
    def generate_data(self, split, num_samples):
        ds = load_dataset(self.config["dataset_name"], streaming=True, split=split)
        counter = 0
        for sample in iter(ds):
            if counter >= num_samples:
                break
            yield {"text": self.format_sample(sample)}
            counter += 1
    
    def load_datasets(self):
        train_dataset = Dataset.from_generator(lambda: self.generate_data("train", self.config["num_train_samples"]))
        val_dataset = Dataset.from_generator(lambda: self.generate_data("train", self.config["num_val_samples"]))
        return train_dataset, val_dataset
    
    def get_training_arguments(self):
        return TrainingArguments(
            output_dir=self.config["output_dir"],
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            optim=self.config["optim"],
            save_steps=self.config["save_steps"],
            logging_steps=self.config["logging_steps"],
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            fp16=self.config["fp16"],
            bf16=self.config["bf16"],
            max_grad_norm=self.config["max_grad_norm"],
            max_steps=self.config["max_steps"],
            warmup_ratio=self.config["warmup_ratio"],
            group_by_length=self.config["group_by_length"],
            evaluation_strategy="steps",
            eval_steps=100,
            report_to="all",
        )
    
    def get_trainer(self):
        peft_config = LoraConfig(
            lora_alpha=self.config["lora_alpha"],
            r=self.config["lora_r"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        return SFTTrainer(
            model=self.model,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val,
            peft_config=peft_config,
            tokenizer=self.tokenizer,
            args=self.training_args,
        )
    
    def train(self):
        self.trainer.train()
    
    def save_model(self):
        self.trainer.model.save_pretrained(self.config["new_model_path"])
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        final_model = PeftModel.from_pretrained(base_model, self.config["new_model_path"])
        final_model = final_model.merge_and_unload()
        final_model.save_pretrained(self.config["final_model_path"])
        self.tokenizer.save_pretrained(self.config["final_model_path"])
    
    def push_to_hub(self):
        self.model.push_to_hub(self.config["hub_repo"])
        self.tokenizer.push_to_hub(self.config["hub_repo"])

if __name__ == "__main__":
    trainer = LlamaTrainer()
    trainer.train()
    trainer.save_model()
    trainer.push_to_hub()
