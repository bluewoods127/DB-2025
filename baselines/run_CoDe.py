import os
import json
import time
from pathlib import Path
from argparse import ArgumentParser
import random
from diffusers import DDIMScheduler
import scheduling_ddim_extended
from pipelines.pipeline_stable_diffusion_CoDe import StableDiffusionPipelineCoDe
import numpy as np
import torch
import rewards

device = "cuda" if torch.cuda.is_available() else "cpu"
inference_dtype = torch.float32


def main(args):
    print(args)
    total_start = time.time()  # <-- Start total time
    
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
        test_prompts = json.load(f)

    if args.reward_fn == 'hps':
        reward_fn = rewards.hps_score(inference_dtype=inference_dtype, device=device)
    elif args.reward_fn == 'pick':
        reward_fn = rewards.PickScore(inference_dtype=inference_dtype, device=device)
    elif args.reward_fn == 'aesthetic':
        aesthetic_fn = rewards.aesthetic_score(torch_dtype=inference_dtype, device=device)
        reward_fn = lambda images, prompts: aesthetic_fn(images, prompts)
    elif args.reward_fn == 'clip':
        clip_fn = rewards.clip_score(inference_dtype=inference_dtype, device=device)
        reward_fn = lambda images, prompts: 20 * clip_fn(images, prompts)
    elif args.reward_fn == 'imagereward':
        image_reward_fn = rewards.ImageReward(inference_dtype=inference_dtype, device=device)
        reward_fn = lambda images, prompts: (image_reward_fn(images, prompts))
    elif args.reward_fn == 'multi':
        vila_fn = rewards.reward_vila
        image_reward = rewards.ImageReward(inference_dtype=inference_dtype, device=device)
        image_reward_fn = lambda images, prompts: (image_reward(images, prompts))
    elif args.reward_fn == 'vila':
        reward_fn = rewards.reward_vila

    pipe = StableDiffusionPipelineExtended.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        cache_dir="/scratch/user/fatemehdoudi/StableDiffusion/dpok/.cache/")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    indices = np.arange(len(test_prompts))
    #[2, 3, 5, 7, 8, 9, 11, 13, 15, 17, 19, 20, 30, 31, 36, 37, 38, 39, 40, 42, 44, 48, 53, 55, 57][::-1]
    if args.is_rev:
        indices=indices[::-1]
    prompts = [test_prompts[i] for i in indices]

    for prompt in prompts:
        prompt_dir = run_dir / prompt.replace(" ", "_").replace("/", "_")
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            image_path = prompt_dir / f"seed{seed}.png"
            if image_path.exists():
                print(f"[Skip] {image_path} already exists.")
                continue

            start = time.time()  # <-- Start image time
            if args.reward_fn == 'multi':
                image = pipe.forward_collect_traj_ddim(
                    prompt=prompt, eta=1.0, n_samples=args.n_samples,
                    block_size=args.block_size, reward_fn_name='multi',
                    ir_reward_fn=image_reward_fn, vila_reward_fn=vila_fn,
                    ir_weight=args.ir_weight)
            else:
                image = pipe.forward_collect_traj_ddim(
                    prompt=prompt, eta=1.0, reward_fn=reward_fn,
                    n_samples=args.n_samples, block_size=args.block_size,
                    reward_fn_name=args.reward_fn)

            image[0].save(image_path)
            duration = time.time() - start  # <-- End image time
            print(f"[Time] {image_path} generated in {duration:.2f} sec")

    total_duration = time.time() - total_start  # <-- End total time
    print(f"[Total Time] Run completed in {total_duration/60:.2f} minutes")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--reward_fn", default="aesthetic")
    parser.add_argument("--ir_weight", type=float, default=0.5)
    parser.add_argument("--n_samples", default=64, type=int)
    parser.add_argument("--n_seeds", default=2, type=int)
    parser.add_argument("--block_size", default=5, type=int)
    parser.add_argument("--prompt_file", type=str, default="dataset/all_prompts.json")
    parser.add_argument("--is_rev", action="store_true")
    parser.add_argument("--run_folder", type=str, default=None, help="Use this folder as run output if provided")
    main(parser.parse_args())
