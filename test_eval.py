import os
import torch
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO
import pickle
import ast
import functools
from transformers import AutoProcessor, AutoModel
from diffusers import UNet2DConditionModel, DDIMScheduler
from pipeline_stable_diffusion_multiobj import StableDiffusionPipelineMultiObj
import ImageReward as imagereward
import utils
import argparse

# === Argument parser ===
parser = argparse.ArgumentParser(description="Evaluate reward scores from multi-object diffusion outputs.")
parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Path to pretrained Stable Diffusion model.")
parser.add_argument("--non_ema_revision", type=str, default=None, help="Optional revision tag for non-EMA UNet weights.")
parser.add_argument("--eval_trainset", type=int, default=0, help="If 1, evaluate on train prompts; otherwise on test prompts.")
parser.add_argument("--mix_rate", type=str, default="[0.25,0.25,0.5]", help="Weighting of reward components.")
parser.add_argument("--t2i_path", type=str, required=True, help="Path to LoRA weights for image-text alignment fine-tuned UNet (e.g. ImageReward).")
parser.add_argument("--vila_path", type=str, required=True, help="Path to LoRA weights for aesthetic score fine-tuned UNet (e.g. VILA).")
parser.add_argument("--hp_path", type=str, required=True, help="Path to LoRA weights for human preference fine-tuned UNet (e.g. PickScore).")
parser.add_argument("--gen_img_savepath", type=str, default="gen_figures/", help="Directory to save generated images.")
parser.add_argument("--eval_savepath", type=str, default="multiObjEval/", help="Directory to save evaluation reward scores.")
parser.add_argument("--num_seeds", type=int, default=30, help="Number of random seeds to use for generation per prompt.")
args = parser.parse_args()

# === Load models ===
pipe = StableDiffusionPipelineMultiObj.from_pretrained(
    args.pretrained_model_name_or_path, torch_dtype=torch.float32
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

unet1 = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)
unet1.to("cuda")
unet2 = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
)
unet2.to("cuda")

pipe.unet.load_attn_procs(args.t2i_path)
unet1.load_attn_procs(args.vila_path)
unet2.load_attn_procs(args.hp_path)

# === Load reward models ===
weight_dtype = torch.float32
image_reward = imagereward.load("ImageReward-v1.0")
image_reward.requires_grad_(False)
image_reward.to("cuda", dtype=weight_dtype)

model_handle = 'https://tfhub.dev/google/vila/image/1'
model = hub.load(model_handle)
predict_fn = model.signatures['serving_default']

processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
hp_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to("cuda")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def calculate_reward_ir(image_pil, prompt):
    score, _ = utils.image_reward_get_reward(image_reward, image_pil, prompt, weight_dtype)
    return score.cpu().squeeze().item()

def reward_vila(img):
    with BytesIO() as byte_stream:
        img.save(byte_stream, format="PNG")
        img_bytes = byte_stream.getvalue()
    score = predict_fn(tf.constant(img_bytes))['predictions'].numpy()[0][0] * 4 - 2
    return score

def reward_hp(prompt, images):
    inputs = processor(images=images, text=prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        image_emb = hp_model.get_image_features(**inputs).float()
        text_emb = hp_model.get_text_features(**inputs).float()
        image_emb /= image_emb.norm(dim=-1, keepdim=True)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        score = hp_model.logit_scale.exp() * (text_emb @ image_emb.T)[0]
    return score.item()

# === Prompts ===
train_prompts = ["A red colored car.", "A black colored dog."]
test_prompts = ["A white colored dog.", "A purple colored dog."]
prompts = train_prompts if args.eval_trainset == 1 else test_prompts

# === Output path ===
mix_rate = ast.literal_eval(args.mix_rate)
mix_rate = [x / sum(mix_rate) for x in mix_rate]
eval_savepath = os.path.join(
    args.eval_savepath,
    "TrainPrompts" if args.eval_trainset else "TestPrompts",
    f"t2i{mix_rate[0]}_vila{mix_rate[1]}_hp{mix_rate[2]}"
)
os.makedirs(eval_savepath, exist_ok=True)
gen_img_savepath = f"gen_figures/DiffusionSoup_t2i{mix_rate[0]}_vila{mix_rate[1]}_hp{mix_rate[2]}/mix_average/shortdrawbench/{'trainPrompts' if args.eval_trainset else 'testPrompts'}/"

# === Evaluation ===
t2i_scores, vila_scores, hp_scores = [], [], []
for j, prompt in enumerate(prompts):
    subdir = os.path.join(gen_img_savepath, str(j))
    os.makedirs(subdir, exist_ok=True)
    for i in range(args.num_seeds):
        set_seed(i)
        images = pipe.forward_ddim_multi_obj(prompt=prompt, unet_obj2=[unet1, unet2], eta=1.0, mix_rate=mix_rate).images
        img = images[0]
        img.save(os.path.join(subdir, f"seed_{i}.png"))
        t2i_scores.append(calculate_reward_ir(img, prompt))
        vila_scores.append(reward_vila(img))
        hp_scores.append(reward_hp(prompt, images))
    with open(os.path.join(eval_savepath, f"t2i_scores_prompt_{j}.pkl"), 'wb') as f:
        pickle.dump(t2i_scores, f)
    with open(os.path.join(eval_savepath, f"vila_scores_prompt_{j}.pkl"), 'wb') as f:
        pickle.dump(vila_scores, f)
    with open(os.path.join(eval_savepath, f"hp_scores_prompt_{j}.pkl"), 'wb') as f:
        pickle.dump(hp_scores, f)

with open(os.path.join(eval_savepath, "t2i_scores.pkl"), 'wb') as f:
    pickle.dump(t2i_scores, f)
with open(os.path.join(eval_savepath, "vila_scores.pkl"), 'wb') as f:
    pickle.dump(vila_scores, f)
with open(os.path.join(eval_savepath, "hp_scores.pkl"), 'wb') as f:
    pickle.dump(hp_scores, f)
