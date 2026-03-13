import os
import yaml
import torch
import argparse
import warnings
import logging
from unsloth import FastLanguageModel

# 🔇 Hide noisy library warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# 🚦 Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_benchmark(override_adapter=None):
    # 1. Load Config
    with open("configs/benchmark_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Use CLI override if provided, otherwise use config file
    adapter_path = override_adapter if override_adapter else config["adapter_path"]

    # 2. Benchmarking Loop
    results = []
    
    print(f"🚀 Loading Aligned Model from {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path,
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
        
        # Clean up response (extract only the assistant's part)
        clean_response = response.split("assistant")[-1].strip()
        results.append({"prompt": prompt, "response": clean_response})
        print(f"✅ Generated Response.")

    # 3. Save Report
    os.makedirs("evaluation", exist_ok=True)
    report_path = "evaluation/benchmark_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"MODEL: {adapter_path}\n")
        f.write("=" * 50 + "\n")
        for res in results:
            f.write(f"PROMPT: {res['prompt']}\n")
            f.write(f"ALIGNED RESPONSE: {res['response']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"🏆 Benchmark complete! Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO Model Benchmarks")
    parser.add_argument("--adapter", type=str, help="Path to LoRA adapter (local or HF ID)")
    args = parser.parse_args()

    run_benchmark(override_adapter=args.adapter)
