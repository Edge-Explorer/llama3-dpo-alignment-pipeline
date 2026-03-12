# 🦙 Llama-3 DPO Alignment Pipeline

A modular, production-ready pipeline for fine-tuning Llama-3 8B using **Direct Preference Optimization (DPO)** — the same technique used to align GPT-4, Claude, and Gemini to human preferences.

---

## 🧠 What Is This?

This project trains a Llama-3 8B model to prefer "good" responses over "bad" ones using the **Intel Orca DPO Pairs** dataset. Instead of just predicting the next token (like standard SFT), DPO teaches the model to **rank responses** — making it more helpful, less harmful, and more aligned with human intent.

**Core formula:**
```
DPO Loss = -log σ(β · (log π(chosen) - log π(rejected)) - (log π_ref(chosen) - log π_ref(rejected)))
```
- `β` controls how strongly the model should deviate from its base behavior (we use `0.1`)
- Higher `rewards/margin` in WandB = model is learning to prefer chosen over rejected ✅

---

## 🗂️ Project Structure

```
llama3-dpo-alignment-pipeline/
├── training/
│   ├── Training_Colab.ipynb        # Original Colab notebook (archived)
│   └── Training_llama3_dpo.ipynb   # Active training notebook
├── data/
│   └── orca_rlhf.jsonl             # Local DPO dataset (chosen/rejected pairs)
├── configs/                        # (Planned) YAML training configs
├── scripts/                        # (Planned) Modular Python training scripts
├── README.md
└── LICENSE
```

---

## ⚙️ Tech Stack

| Component | Library | Purpose |
|:--|:--|:--|
| Base Model | `unsloth/llama-3-8b-Instruct-bnb-4bit` | 4-bit quantized Llama-3 |
| Efficiency | `Unsloth 2026.3.4` | 2x faster training, 60% less VRAM |
| Adapter | `LoRA (r=64)` via PEFT | Train only 2% of parameters |
| Alignment | `DPO` via TRL | Preference-based alignment |
| Tracking | `WandB` | Real-time loss & reward curves |
| Hardware | `Tesla T4 (15GB)` | Google Colab free tier |

---

## 🚨 Training Environment: Known Issue & Recommendation

### The Core Problem
Google Colab's free tier runs **Python 3.12** with **Transformers 5.0.0** pre-installed. The `trl` library (which provides `DPOTrainer`) and `Unsloth`'s patched trainer are currently **not fully compatible** with this combination, causing cascading `ImportError`, `TypeError`, and `AttributeError` issues regardless of which `trl` version is installed.

### The Dependency Conflict Chain
```
Colab pre-installs: transformers==5.0.0
↓
trl 0.8.6 (needed for Unsloth) → broke with transformers 5.0
↓
trl latest → DPOConfig API changed, Unsloth patch breaks
↓
Unsloth compiled cache → hardcoded expectations of specific trl+transformers versions
```

### ✅ Recommended Alternatives (Pick One)

| Platform | GPU | Cost | Why It Works |
|:--|:--|:--|:--|
| **Kaggle Notebooks** | T4 16GB | 🆓 Free (30h/week) | Older, stable Python env. Full pip control. **Best free option.** |
| **Lightning AI** | T4 (4h/day) | 🆓 Free tier | Clean Ubuntu container, install exact versions |
| **Vast.ai** | Any GPU | 💰 ~$0.20/hr | Full Docker control, no pre-installed conflicts |
| **RunPod** | A100/H100 | 💰 ~$0.50/hr | Clean PyTorch images, fast training |
| **Google Colab Pro** | A100 | 💰 $10/month | Same environment, but longer sessions |

### 🏆 Recommendation: **Kaggle**
Kaggle is the closest free alternative to Colab. It:
- Has a T4 GPU (same as Colab free)
- Has a **stable, older Python environment** that doesn't conflict with `trl 0.8.x`
- Supports uploading datasets directly
- Has no pre-installed "bleeding edge" packages that break things

---

## 🚀 Quick Start (Kaggle)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Enable GPU: Settings → Accelerator → **GPU T4 x2**
3. Upload `orca_rlhf.jsonl` via the **Data** panel
4. Use the cells from `Training_llama3_dpo.ipynb`

---

## 📊 Training Config

```yaml
model: unsloth/llama-3-8b-Instruct-bnb-4bit
lora_r: 64
lora_alpha: 64
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
beta: 0.1
learning_rate: 5e-6
batch_size: 2
gradient_accumulation: 4   # effective batch = 8
max_length: 1024
max_prompt_length: 512
epochs: 1
optimizer: paged_adamw_8bit
dataset: Intel/orca_dpo_pairs (1000 samples)
```

---

## 📈 What to Watch in WandB

| Metric | Expected Trend | Meaning |
|:--|:--|:--|
| `rewards/margins` | ⬆️ Increasing | Model learns to prefer `chosen` |
| `rewards/chosen` | ⬆️ Increasing | Model assigns higher reward to good responses |
| `rewards/rejected` | ⬇️ Decreasing | Model assigns lower reward to bad responses |
| `loss` | ⬇️ Decreasing | Model is converging |

---

## 🗺️ Roadmap

- [x] Project scaffold & structure
- [x] Dataset preparation (`orca_rlhf.jsonl`)
- [x] Notebook-based DPO training pipeline
- [ ] Resolve training environment (Colab vs Kaggle)
- [ ] Successful DPO training run
- [ ] Save & push LoRA adapter to HuggingFace Hub
- [ ] Inference & evaluation notebook
- [ ] Convert to modular `scripts/train_dpo.py`
- [ ] YAML config system (`configs/dpo_config.yaml`)
- [ ] Automated benchmarking (MT-Bench / TruthfulQA)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
