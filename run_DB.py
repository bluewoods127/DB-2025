import sys
sys.path.append("..")
import ImageReward as imagereward
import utils
import os
import json
from pathlib import Path
from argparse import ArgumentParser
import random
from diffusers import DDIMScheduler
from diffusers import UNet2DConditionModel
from pipeline_stable_diffusion_multiobj import StableDiffusionPipelineMultiObj
import numpy as np
import torch
import copy
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from io import BytesIO
import functools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

device = "cuda" if torch.cuda.is_available() else "cpu"
inference_dtype = torch.float32

weight_dtype = torch.float32

def _calculate_reward_ir(image_reward, image_pil, prompts):
    blip_reward, _ = utils.image_reward_get_reward(
      image_reward, image_pil, prompts, weight_dtype
    )
    return blip_reward.cpu().squeeze(0).squeeze(0)


def reward_vila(vila_model, img):
    with BytesIO() as byte_stream:
        img.save(byte_stream, format="PNG") 
        img_bytes = byte_stream.getvalue() 
    prediction = vila_model(tf.constant(img_bytes))
    score = prediction['predictions']
    numpy_array = score.numpy()*4-2  #multiply by 4 to scale as image_reward
    return numpy_array


def set_seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def set_seed_tf(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(args):
    print(args)
    image_reward = imagereward.load("ImageReward-v1.0" , download_root=args.cache_dir)
    image_reward.requires_grad_(False)
    image_reward.to('cuda', dtype=weight_dtype)
    calculate_ir = functools.partial(_calculate_reward_ir, image_reward)
    
    model_vila = hub.load('https://tfhub.dev/google/vila/image/1')
    calculate_vila = functools.partial(reward_vila, model_vila.signatures['serving_default'])

    if args.run_folder:
        run_dir = Path(args.run_folder)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = Path("runs")
        base_dir.mkdir(exist_ok=True)
        existing_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        run_id = int(existing_dirs[-1].name) + 1 if existing_dirs else 0
        run_dir = base_dir / f"{run_id:03d}"
        run_dir.mkdir()

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(args.prompt_file, "r") as f:
        prompts_list = json.load(f)

  
    pipe = StableDiffusionPipelineMultiObj.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.unet.load_attn_procs(args.t2i_path)
    
    
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", revision=None)
    unet.to("cuda")
    unet.load_attn_procs(args.vila_path)

    
    for prompt in prompts_list:
        print(prompt)
        prompt_dir = run_dir / prompt.replace(" ", "_").replace("/", "_")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        t2i_scores = []
        vila_scores = []
        for seed in range(args.n_seeds):
            set_seed_torch(seed)
            set_seed_tf(seed)

            image_path = prompt_dir / f"seed{seed}.png"
            if image_path.exists():
                print(f"[Skip] {image_path} already exists.")
                continue

            images = pipe.forward_ddim_multi_obj(prompt=prompt, unet_obj2=unet, eta=1.0, mix_rate=args.ir_weight).images
            images[0].save(image_path)
            
            reward_img = calculate_ir(images[0], prompt)
            vila_score = calculate_vila(images[0])
            
            reward_img, vila_score = reward_img.item(), vila_score[0][0]
            t2i_scores.append(reward_img)
            vila_scores.append(vila_score)
            
        with open(prompt_dir / f"t2i_w_{args.ir_weight}.pkl", 'wb') as file:
            pickle.dump(t2i_scores, file)
        with open(prompt_dir / f"vila_w_{args.ir_weight}.pkl", 'wb') as file:
            pickle.dump(vila_scores, file)


if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument("--ir_weight", type=float, default=0.5)
    parser.add_argument("--n_seeds", default=5, type=int)
    parser.add_argument("--prompt_file", type=str, default="dataset/all_prompts.json")
    parser.add_argument("--run_folder", type=str, default=None)
    parser.add_argument("--t2i_path", type=str, required=True)
    parser.add_argument("--vila_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=".cache")
    
    main(parser.parse_args())
