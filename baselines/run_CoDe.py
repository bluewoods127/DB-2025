import os
import json
from pathlib import Path
from argparse import ArgumentParser
import random
from diffusers import DDIMScheduler
import scheduling_ddim_extended
import pipeline_stable_diffusion_CoDe
from pipeline_stable_diffusion_CoDe import StableDiffusionPipelineCoDe
import numpy as np
import torch
import rewards

device = "cuda" if torch.cuda.is_available() else "cpu"
inference_dtype = torch.float32

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
        base_dir = Path("runs")
        base_dir.mkdir(exist_ok=True)
        existing_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        run_id = int(existing_dirs[-1].name) + 1 if existing_dirs else 0
        run_dir = base_dir / f"{run_id:03d}"
        run_dir.mkdir()

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(args.prompt_file, "r") as f:
        prompt_list = json.load(f)
    
    vila_fn = rewards.reward_vila
    image_reward = rewards.ImageReward(inference_dtype=inference_dtype, device=device)
    image_reward_fn = lambda images, prompts: (image_reward(images, prompts))

    pipe = StableDiffusionPipelineCoDe.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir=args.cache_dir)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    
    for prompt in prompt_list:
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

            images = pipe.forward_collect_traj_ddim( 
                prompt=prompt, eta=1.0, n_samples=args.n_samples,
                block_size=args.block_size, reward_fn_name='multi',
                ir_reward_fn=image_reward_fn, vila_reward_fn=vila_fn,
                ir_weight=args.ir_weight)

            images[0].save(image_path)
            
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
    
    parser.add_argument("--ir_weight", type=float, default=0.5)
    parser.add_argument("--n_samples", default=20, type=int)
    parser.add_argument("--n_seeds", default=2, type=int)
    parser.add_argument("--block_size", default=5, type=int)
    parser.add_argument("--prompt_file", type=str, default="dataset/all_prompts.json")
    parser.add_argument("--run_folder", type=str, default=None, help="Use this folder as run output if provided")
    parser.add_argument("--cache_dir", type=str, default='.cache/')

    main(parser.parse_args())
