import os
import yaml 
import torch
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer
import wandb

# 1. Load Configuration
with open("configs/dpo_config.yaml", "r") as f:
    config= yaml.safe_load(f)

os.environ["PYTORCH_CUDA_ALLOC_CONF"]= "expandable_segments: True"

def train():
    # 2. Initialize Model & Tokenizer
    model, tokenizer= FastLanguageModel.from_pretrained(
        model_name= config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit= config["load_in_4bit"],
    )

    # 3. Add LoRA Adapters
    model= FastLanguageModel.get_peft_model(
        model,
        **config["lora_config"]
    )

    # 4. Prepare Dataset (Formatting for Llama-3)
    def format_dpo_sample(sample):
        return {
            "prompt"  : f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "chosen"  : f"{sample['chosen']}<|eot_id|>",
            "rejected" : f"{sample['rejected']}<|eot_id|>",
        }

    dataset= load_dataset(config["dataset_name"], split="train[:1000]")
    dataset= dataset.map(format_dpo_sample)

    # 5. Patch & Initialize DPO Trainer
    PatchDPOTrainer()

    training_args= DPOConfig(
        **config["dpo_config"]
    )

    trainer= DPOTrainer(
        model= model,
        ref_model= None,
        args= training_args,
        train_dataset= dataset,
        tokenizer= tokenizer,
    )

    # 6. Execution
    print("Starting Professional DPO Training...")
    trainer.train()

    # 7. Save the Result
    output_path = "models/llama3_dpo_adapter_local"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✅ Success! Aligned adapter saved to {output_path}")


if __name__ == "__main__":
    train()