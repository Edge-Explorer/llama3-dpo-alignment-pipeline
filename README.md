# 🦙 Llama-3 DPO Alignment Pipeline

> A professional, modular pipeline to fine-tune **Llama-3 8B** using **Direct Preference Optimization (DPO)** — from training to inference to benchmarking.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-orange)](https://github.com/unslothai/unsloth)
[![TRL](https://img.shields.io/badge/TRL-DPO-green)](https://github.com/huggingface/trl)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Karan6124%2Fllama3--8b--dpo--orca--adapter-yellow)](https://huggingface.co/Karan6124/llama3-8b-dpo-orca-adapter)
[![GitHub](https://img.shields.io/badge/GitHub-Edge--Explorer-black?logo=github)](https://github.com/Edge-Explorer/llama3-dpo-alignment-pipeline)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🚀 What is This Project?

This repository contains a **complete, production-grade pipeline** for aligning a large language model using DPO. Instead of manually patching notebooks together, this project is structured as reusable Python scripts with YAML configurations so **anyone can reproduce the training** with a single command.

### ⚡ The Core Idea
| Concept | What We Did |
|---|---|
| **Base Model** | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| **Alignment Technique** | Direct Preference Optimization (DPO) |
| **Dataset** | `Intel/orca_dpo_pairs` (1,000 samples) |
| **Speed Optimization** | Unsloth (2x faster training) |
| **Memory Optimization** | 4-bit quantization + Gradient Checkpointing |
| **Environment** | Kaggle T4 x2 GPU (Free Tier) |
| **Trained Adapter** | [🤗 Karan6124/llama3-8b-dpo-orca-adapter](https://huggingface.co/Karan6124/llama3-8b-dpo-orca-adapter) |

---

## 🗂️ Project Structure

```
llama3-dpo-alignment-pipeline/
├── 📁 configs/
│   ├── dpo_config.yaml          # All DPO training hyperparameters
│   └── benchmark_config.yaml   # Test prompts & generation settings
├── 📁 scripts/
│   └── train_dpo.py            # The main training engine (reads from configs/)
├── 📁 inference/
│   └── inference.py            # Load adapter and run interactive inference
├── 📁 evaluation/
│   └── benchmark.py            # Compare Base vs. Aligned model side-by-side
├── 📁 training/
│   └── training-llama3-dpo.ipynb  # The original Kaggle notebook
├── 📁 models/                  # (gitignored) Local adapter weights live here
├── pyproject.toml              # Dependency management with uv
└── README.md
```

---

## 🛠️ Setup & Installation

This project uses [**uv**](https://github.com/astral-sh/uv) — the fastest Python package manager. Everything is managed in one command.

### 1. Install `uv`
```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the Repo & Sync Dependencies
```bash
git clone https://github.com/Edge-Explorer/llama3-dpo-alignment-pipeline.git
cd llama3-dpo-alignment-pipeline
uv sync
```

> **Note:** Unsloth requires an NVIDIA GPU to import. For local development, you can write and review code without a GPU. Use Kaggle or Google Colab to actually run the scripts.

---

## 🏋️ Training

All training parameters are in `configs/dpo_config.yaml`. You can tweak learning rates, batch sizes, and sequence lengths without touching any Python code.

```bash
# On Kaggle or a GPU machine:
uv run python scripts/train_dpo.py
```

**Key Hyperparameters (optimized for T4 GPU):**
- `beta`: 0.1 (DPO temperature)
- `learning_rate`: 5e-6
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 8
- `max_length`: 768

---

## 🤖 Inference

Download the trained adapter from Hugging Face and load it locally.

```bash
# Make sure the adapter is in models/llama3_dpo_adapter/
uv run python inference/inference.py
```

Or use it directly in your own code:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Karan6124/llama3-8b-dpo-orca-adapter",
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "Why use DPO over SFT?"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

---

## 📊 Benchmarking

Run the benchmark script on Kaggle to generate a comparison report:

```bash
uv run python evaluation/benchmark.py
```

Results are saved automatically to `evaluation/benchmark_report.txt`.

---

## 🏆 Key Lessons Learned

| Problem | Solution |
|---|---|
| `transformers 5.0.0` broke `trl` in Colab | Switched to Kaggle's stable environment |
| `DPOConfig` not found in old `trl` | Pinned to `trl>=0.12.0` |
| `OutOfMemoryError` on T4 GPU | Reduced batch size to 1, enabled gradient checkpointing |
| Slow training | Unsloth's `PatchDPOTrainer` gave ~2x speedup |
| Messy notebook workflow | Refactored into reusable scripts + YAML configs |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for making LLM fine-tuning incredibly fast
- [TRL by HuggingFace](https://github.com/huggingface/trl) for the DPO implementation
- [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) for the training dataset
- [Kaggle](https://kaggle.com) for the free GPUs that made this possible
