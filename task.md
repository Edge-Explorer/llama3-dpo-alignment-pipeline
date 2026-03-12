# Project Roadmap: Llama-3 DPO Alignment

- [x] **Phase 1: Foundation (Environment & Model Loading)**
    - [x] Initial setup and library installation.
    - [x] Corrected version conflicts (Unsloth vs. Transformers).
- [x] **Phase 2: Data Engineering (DPO Formatting & Local Loading)**
    - [x] Formatted Orca RLHF pairs for Llama-3 Instruct template.
- [x] **Phase 3: Training Engine (Llama-3 DPO Training)**
    - [x] Successfully trained on Kaggle (T4 x2) for 125 steps.
    - [x] Implemented VRAM optimizations (expandable_segments, batch=1).
- [x] **Phase 4: Evaluation & Inference (Verify Alignment)**
    - [x] Performed inference and verified model behavior.
- [ ] **Phase 5: Export & Serialization (Push to Hub / GGUF)**
    - [ ] Clean up notebook code into modular scripts.
    - [ ] Create GGUF version for local use.
