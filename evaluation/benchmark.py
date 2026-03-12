import os
import yaml
import torch
from unsloth import FastLanguageModel

# 🚦 Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_benchmark():
    # 1. Load Config
    with open("configs/benchmark_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Benchmarking Loop (This is a simplified version)
    results = []
    
    # NOTE: In a real run on Kaggle, we would load Base, 
    # generate, clear memory, then load Aligned.
    # For now, let's just write the skeleton for the Aligned model.
    print(f"🚀 Loading Aligned Model from {config['adapter_path']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["adapter_path"],
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    print("📊 Evaluating Prompts...")
    for prompt in config["test_prompts"]:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        print(f"❓ Prompt: {prompt}")
        outputs = model.generate(**inputs, **config["generation_params"])
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        results.append({"prompt": prompt, "response": response.split("assistant")[-1].strip()})
        print(f"✅ Generated Response.")

    # 3. Save Report
    with open("evaluation/benchmark_report.txt", "w", encoding="utf-8") as f:
        for res in results:
            f.write(f"PROMPT: {res['prompt']}\n")
            f.write(f"ALIGNED RESPONSE: {res['response']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"🏆 Benchmark complete! Report saved to evaluation/benchmark_report.txt")

if __name__ == "__main__":
    run_benchmark()
