# coding=utf-8
# Copyright 2024 The Google Research Authors.

"""Extension of diffusers.StableDiffusionPipeline."""

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
import torch
import copy
from PIL import Image




class StableDiffusionPipelineCoDe(StableDiffusionPipeline):
  """Extension of diffusers.StableDiffusionPipeline."""
  def _decode(self, latents):
    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    image = self.image_processor.postprocess(image, output_type='pt', do_denormalize=do_denormalize)
    return image

  # Return the full trajectory
  def forward_collect_traj_ddim(
      self,
      prompt = None,
      height = None,
      width = None,
      num_inference_steps = 50,
      guidance_scale = 7.5,
      negative_prompt = None,
      num_images_per_prompt = 1,
      eta = 1.0,
      generator = None,
      latents = None,
      prompt_embeds = None,
      negative_prompt_embeds = None,
      output_type = 'pil',
      return_dict = True,
      callback = None,
      callback_steps = 1,
      cross_attention_kwargs = None,
      is_ddp = False,
      unet_copy=None,
      n_samples=2,
      reward_fn=None,
      block_size=5,
      reward_fn_name=None,
      ir_reward_fn=None,
      vila_reward_fn=None,
      ir_weight = 1.0,
  ):
   
    if is_ddp:
      height = (
          height or self.unet.module.config.sample_size * self.vae_scale_factor
      )
      width = (
          width or self.unet.module.config.sample_size * self.vae_scale_factor
      )
    else:
      height = height or self.unet.config.sample_size * self.vae_scale_factor
      width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )

    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    if is_ddp:
      num_channels_latents = self.unet.module.in_channels
    else:
      num_channels_latents = self.unet.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    latents_list = []
    log_prob_list = []
    latents_list.append(latents.detach().clone().cpu())
    # 6. Prepare extra step kwargs.
    # TODO: Logic should ideally just be moved out of the pipeline  # pylint: disable=g-bad-todo
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = (
        len(timesteps) - num_inference_steps * self.scheduler.order
    )
    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for i, t in enumerate(timesteps):
        
        prev_timestep = (t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)
        prev_timestep = torch.clamp(prev_timestep, 0, self.scheduler.config.num_train_timesteps - 1)
        latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
        

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample


        if do_classifier_free_guidance:
          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
          noise_pred = noise_pred_uncond + guidance_scale * (
              noise_pred_text - noise_pred_uncond
          )
        if ((t > timesteps[-1]) and (i % block_size == 0)): # or (i == len(timesteps) - 2):
            gen_rewards = []
            gen_samples = []
            with torch.no_grad():
                for iter in range(n_samples):
                    latent_temp = latents.detach().clone()
                    latent_temp = self.scheduler.step(noise_pred, t, latent_temp, **extra_step_kwargs).prev_sample
                    latent_temp_input = torch.cat([latent_temp] * 2) if do_classifier_free_guidance else latent_temp
                    latent_temp_input = self.scheduler.scale_model_input(latent_temp_input, timesteps[i + 1])
                    noise_pred_temp = self.unet(latent_temp_input, timesteps[i + 1], encoder_hidden_states=prompt_embeds,
                                                cross_attention_kwargs=cross_attention_kwargs,).sample
                    if do_classifier_free_guidance:
                        noise_pred_temp_uncond, noise_pred_temp_text = noise_pred_temp.chunk(2)
                        noise_pred_temp = noise_pred_temp_uncond + guidance_scale * (noise_pred_temp_text - noise_pred_temp_uncond)
                    pred_original_temp = self.scheduler.step(noise_pred_temp, timesteps[i + 1],
                                                             latent_temp, **extra_step_kwargs).pred_original_sample


                    decoded = self._decode(pred_original_temp)
                    if reward_fn_name == 'vila':
                        if isinstance(decoded, torch.Tensor):
                            decoded = decoded.detach().cpu().float().clamp(0, 1)  # ensure in [0, 1]
                            decoded = (decoded * 255).round().byte()              # (B, C, H, W) -> uint8
                            decoded = decoded.permute(0, 2, 3, 1).numpy()          # (B, H, W, C)
                        decoded = [Image.fromarray(img) for img in decoded]       # list of PIL Images
                        reward = reward_fn([decoded[0]], prompt).unsqueeze(0)
                        
                    elif reward_fn_name == 'multi':
                        ir_reward = ir_reward_fn(decoded, prompt).unsqueeze(0)
                        
                        if isinstance(decoded, torch.Tensor):
                            decoded = decoded.detach().cpu().float().clamp(0, 1)  # ensure in [0, 1]
                            decoded = (decoded * 255).round().byte()              # (B, C, H, W) -> uint8
                            decoded = decoded.permute(0, 2, 3, 1).numpy()          # (B, H, W, C)
                        decoded = [Image.fromarray(img) for img in decoded]       # list of PIL Images
                        vila_reward = vila_reward_fn([decoded[0]], prompt)
                        vila_reward = vila_reward.squeeze()                  
                        vila_reward = vila_reward.view_as(ir_reward)
                        vila_reward = vila_reward.to(ir_reward.device)

                        reward = ir_weight * ir_reward + (1 - ir_weight) * vila_reward
                    
                    else:
                        reward = reward_fn(decoded, prompt).unsqueeze(0)
                    
                    gen_rewards.append(reward)
                    gen_samples.append(latent_temp.unsqueeze(0).detach().clone())
                    del decoded, reward, latent_temp_input, noise_pred_temp, pred_original_temp
                    torch.cuda.empty_cache()
            
            # select_ind = torch.max(torch.cat(gen_rewards), dim=0)[1]
            # gen_samples = torch.cat(gen_samples, dim=0)
            # gen_samples = gen_samples.permute(1,0,2,3,4)
            # latents = torch.cat([x[select_ind[idx]].unsqueeze(0) for idx, x in enumerate(gen_samples)],dim=0).detach().clone()

            rewards_tensor = torch.cat(gen_rewards, dim=0)  # shape: (n_samples, batch_size)
            rewards_tensor = rewards_tensor.view(n_samples, -1).T  # shape: (batch_size, n_samples)
            select_ind = torch.argmax(rewards_tensor, dim=1) 

            gen_samples_tensor = torch.cat(gen_samples, dim=0)
            gen_samples_tensor = gen_samples_tensor.permute(1, 0, 2, 3, 4)
            selected_latents = torch.stack([
                gen_samples_tensor[b, select_ind[b]] for b in range(select_ind.shape[0])
            ], dim=0)

            latents = selected_latents.detach().clone()


            del gen_rewards, gen_samples, select_ind
            torch.cuda.empty_cache()
        else:
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                

        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
          progress_bar.update()
          if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

    if output_type == "latent":
      image = latents
    elif output_type == "pil":
      latents = latents.detach()
      latents = latents.to(prompt_embeds.dtype)
      image = self._decode(latents)

      if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
      image = self.numpy_to_pil(image)

    else:
      image = self._decode(latents)

    if (
        hasattr(self, "final_offload_hook")
        and self.final_offload_hook is not None
    ):
      self.final_offload_hook.offload()

    return image
