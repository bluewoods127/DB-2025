print('daaaaaah!')
import os
import json
from pathlib import Path
from argparse import ArgumentParser
import random
from diffusers import DDIMScheduler
import scheduling_ddim_extended
import pipeline_stable_diffusion_RGG
from pipeline_stable_diffusion_RGG import StableDiffusionPipelineRGG
import numpy as np
import torch
import rewards
import time
from load_vila import load_vila_model
import tensorflow as tf
import pickle



device = "cuda" if torch.cuda.is_available() else "cpu"
inference_dtype = torch.float32


def create_unique_run_dir(base_dir):
    base_dir.mkdir(exist_ok=True)
    for _ in range(100):
        existing_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        run_id = int(existing_dirs[-1].name) + 1 if existing_dirs else 0
        run_dir = base_dir / f"{run_id:03d}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            time.sleep(random.uniform(0.1, 0.5))
    raise RuntimeError("Failed to create a unique run directory after 100 attempts.")

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

    if args.run_folder:
        run_dir = Path(args.run_folder)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_unique_run_dir(Path("runs"))

    # Save args
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(args.prompt_file, "r") as f:
        prompt_list = json.load(f)
    
    image_reward = rewards.ImageReward(inference_dtype=inference_dtype, device=device)
    image_reward_fn = lambda images, prompts: image_reward(images, prompts)
    
    vila_fn = rewards.reward_vila
    vila_reward_net , vila_reward_net_states = load_vila_model('./vila/checkpoints/vila_rank_tuned/')
    
    
    pipe = StableDiffusionPipelineRGG.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir=args.cache_dir)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    for prompt in prompt_list:
        prompt_dir = run_dir / prompt.replace(" ", "_").replace("/", "_")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        t2i_scores = []
        vila_scores = []
        
        for seed in range(args.n_sample):
            set_seed_torch(seed)
            set_seed_tf(seed)

            seed_file = prompt_dir / f"seed{seed}.png"
            if seed_file.exists():
                print(f"[Skip] seed{seed} already exists for prompt '{prompt}', skipping.")
                continue

            images = pipe.forward_collect_traj_ddim(
                prompt=prompt, eta=1.0,
                kl_coeff=args.kl_coeff,
                is_tempering=args.is_tempering,
                gamma=args.gamma,
                reward_fn_name='multi',
                ir_reward_fn=image_reward_fn,
                vila_reward_net=vila_reward_net,
                vila_reward_net_states=vila_reward_net_states,
                ir_weight=args.ir_weight,
                num_inference_steps=args.n_inference,
                do_guidance=args.do_guidance,
                grad_norm=args.grad_norm,
            )

            images[0].save(prompt_dir / f"seed{seed}.png")
            
            reward_img = image_reward_fn(images[0], prompt)
            vila_score = vila_fn(images[0])
            
            reward_img, vila_score = reward_img.item(), vila_score[0][0]
            t2i_scores.append(reward_img)
            vila_scores.append(vila_score)
            
        with open(prompt_dir / f"t2i_w_{args.ir_weight}.pkl", 'wb') as file:
            pickle.dump(t2i_scores, file)
        with open(prompt_dir / f"vila_w_{args.ir_weight}.pkl", 'wb') as file:
            pickle.dump(vila_scores, file)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--ir_weight", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--is_tempering", action="store_true")
    parser.add_argument("--n_sample", default=64, type=int)
    parser.add_argument("--prompt_file", type=str, default="dataset/all_prompts.json")
    parser.add_argument("--n_inference", default=50, type=int)
    parser.add_argument("--run_folder", type=str, default=None, help="Manually specify run directory")
    parser.add_argument("--do_guidance", default=1.0, type=float)
    parser.add_argument("--grad_norm", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=".cache")

    

    main(parser.parse_args())
