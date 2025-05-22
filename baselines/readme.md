
# DB-2025: Baseline Execution Details

This document provides a summary of how to execute each baseline model used in the DB-2025 study, based on the commands defined in `run.sh`.

---

## 🧪 Baselines Included

- **Rewarded Soup (RS)**  
- **Compositional Decoding (CoDe)**  
- **Rewarded Gradient Guidance (RGG)**

---

## ⚙️ Execution Environments

- RS and CoDe use the **same environment** as DB (referred to as `DB_env`).
- RGG uses a **separate environment** (`jax_torch_env`) to isolate JAX dependencies.

---

## ▶️ Execution Commands

### 🎯 Rewarded Soup (RS)

```bash
source activate DB_env
python3 -u run_RS.py \
  --cache_dir ".cache/" \
  --ir_weight 0.2 \
  --prompt_file "dataset/test_prompts.json" \
  --t2i_path 'path_to_checkpoint/pytorch_lora_weights.bin' \
  --vila_path 'path_to_checkpoint/pytorch_lora_weights.bin'
```

---

### 🧩 Compositional Decoding (CoDe)

```bash
source activate DB_env
python3 -u run_CoDe.py \
  --cache_dir ".cache/" \
  --ir_weight 0.2 \
  --prompt_file "dataset/test_prompts.json" \
  --n_samples 20 \
  --block_size 5
```

---

### 🌌 Rewarded Gradient Guidance (RGG)

```bash
source activate jax_torch_env
python3 -u run_guidance.py \
  --kl_coeff 0.1 \
  --reward_fn 'multi' \
  --is_tempering \
  --gamma 0.024 \
  --grad_norm \
  --ir_weight 0.2
```

> ⚠️ Make sure the environment and all dependencies (Torch, JAX, TensorFlow, etc.) are installed before running.

---

## 📂 Notes

- Checkpoints must be correctly placed at the paths provided in the arguments.
- `run.sh` can be used as a single launcher to sequentially run all baselines.
