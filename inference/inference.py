import os
import torch
from unsloth import FastLanguageModel

# Memory Optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"]= "expandable_segments: True"

def run_inference(adapter_path="models/llama3_dpo_adapter"):
    # 1. Load Model + Adapter
    model, tokenizer= FastLanguageModel.from_pretrained(
        model_name= adapter_path,
        max_seq_length= 2048,
        load_in_4bit= True,
    )

    # 2. Switch to Fast Inference Mode
    FastLanguageModel.for_inference(model)

    # 3. Chat Template
    messages= [
        {"role": "user", "content": "Explain the concept of DPO in simple terms."},
    ]

    inputs= tokenizer.apply_chat_template(
        messages,
        tokenize= True,
        add_generation_prompt= True,
        return_tensors= "pt",
    ).to("cuda")

     # 4. Generate (No extra space at the start!)
    print("Thinking...")
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 512,
        use_cache = True
    )
    # 5. Decode
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("\n--- RESPONSE ---\n")
    print(response.split("assistant")[-1].strip())
if __name__ == "__main__":
    # Change the path if yours is different
    path = "models/llama3_dpo_adapter" 
    if os.path.exists(path):
        run_inference(path)
    else:
        print(f"Error: Could not find adapter at {path}. Please unzip your adapter there!")