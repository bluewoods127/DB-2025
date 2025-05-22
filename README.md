
# Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models

**Diffusion Blend** addresses the challenge of aligning diffusion models with multiple user-defined objectives (e.g., aesthetics, text-image alignment) at inference time—without additional fine-tuning. It introduces a novel framework for *inference-time multi-preference alignment*, offering flexible control over both **reward trade-offs** and **KL regularization strength**.

### Key Contributions:
- Introduces DB-MPA (Multi-Preference Alignment) and DB-KLA (KL Alignment).
- Enables blending of backward diffusion processes from fine-tuned models.
- Achieves near-oracle performance while maintaining efficiency.
- Eliminates the need for per-preference fine-tuning.

### 🌈 Visual Overview

<p align="center">
  <img src="assets/db_mpa_overview.png" alt="Diffusion Blend Overview" width="600"/>
</p>

*Figure: DB-MPA blends denoisers from fine-tuned models at inference time to align with user-specified reward vectors.*

> See full paper: [`2025_Diffusion_Blend.pdf`](2025_Diffusion_Blend.pdf)


## 📜 License References

- [Google Research DPOK License](https://github.com/google-research/google-research/tree/master/dpok)
- [DAS Krafton License](https://github.com/krafton-ai/DAS/blob/main/das)

---

## ✅ Step 1: Environment Setup

Create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate DB_env
```

> **Note:** The environment defined in `environment.yaml` is used for most parts of the paper.

For the **RGG baseline** (which uses JAX for Vila reward), we created a separate environment to avoid interference with the main models.

---

## 📦 Step 2: Install ImageReward

Install the ImageReward module:

```bash
bash install_image_reward.sh
```

---

## 🏋️ Step 3: Training with DPOK

To fine-tune models for a given `(reward_weight, kl_weight)` setting, use the `train_online_pg.py` script to obtain checkpoints:

```bash
accelerate launch train_online_pg.py \
  --p_batch_size 2 \
  --reward_weight 1000 \
  --kl_weight 0.1 \
  --learning_rate 1e-5 \
  --single_flag 0 \
  --prompt_path ./dataset/test_prompts.json \
  --gradient_accumulation_steps 12 \
  --clip_norm 0.1 \
  --g_batch_size 6 \
  --multi_gpu 1 \
  --max_train_steps 100000 \
  --v_flag 1 \
  --cache_dir '.cache/'
```

---

## 🚀 Step 4: Run DB Code

After training and obtaining checkpoints for each reward model, use the `run_DB.py` scripts to generate images and evaluate performance, depending on the setup and reward functions used.

```bash
python3 -u run_DB.py --cache_dir ".cache/" \
--ir_weight 0.2 \
--prompt_file "dataset/test_prompts.json" \
--t2i_path 'path_to_checkpoint/pytorch_lora_weights.bin' \
--vila_path 'path_to_checkpoint/pytorch_lora_weights.bin'
```
