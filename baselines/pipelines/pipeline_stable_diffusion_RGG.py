# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extension of diffusers.StableDiffusionPipeline."""

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
import torch
from ddim_with_logprob import ddim_prediction_with_logprob, ddim_step_with_mean , get_variance
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
import jax
import jax.numpy as jnp
from praxis import base_layer
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules
import torch.nn.functional as F
import numpy as np
from functools import partial



NestedMap = py_utils.NestedMap

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

_PRE_CROP_SIZE = 272
_IMAGE_SIZE = 224
_MAX_TEXT_LEN = 64
_TEXT_VOCAB_SIZE = 64000

_ZSL_QUALITY_PROMPTS = [
    ['good image', 'bad image'],
    ['good lighting', 'bad lighting'],
    ['good content', 'bad content'],
    ['good background', 'bad background'],
    ['good foreground', 'bad foreground'],
    ['good composition', 'bad composition'],
]


class StableDiffusionPipelineRGG
(StableDiffusionPipeline):
  """Extension of diffusers.StableDiffusionPipeline."""

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
      soft_reward=False,
      kl_coeff=1.0,
      is_tempering=None,
      gamma=0.0,
      ir_reward_fn=None,
      vila_reward_net=None,
      vila_reward_net_states=None,
      ir_weight=1.0,
      reward_fn_name=None,
      reward_fn=None,
      do_guidance=1.0,
      grad_norm=False,
  ):
    # pylint: disable=line-too-long
    r"""Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*): The prompt or prompts to
          guide the image generation. If not defined, one has to pass
          `prompt_embeds` instead.
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
          The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
          The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50): The number of
          denoising steps. More denoising steps usually lead to a higher quality
          image at the expense of slower inference.
        guidance_scale (`float`, *optional*, defaults to 7.5): Guidance scale as
          defined in [Classifier-Free Diffusion
          Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is
          defined as `w` of equation 2. of [Imagen
          Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is
          enabled by setting `guidance_scale > 1`. Higher guidance scale
          encourages to generate images that are closely linked to the text
          `prompt`, usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*): The prompt or
          prompts not to guide the image generation. If not defined, one has to
          pass `negative_prompt_embeds`. instead. If not defined, one has to
          pass `negative_prompt_embeds`. instead. Ignored when not using
          guidance (i.e., ignored if `guidance_scale` is less than `1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1): The number of
          images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0): Corresponds to parameter eta
          (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies
          to [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
          One or a list of [torch
          generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
          to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*): Pre-generated noisy latents,
          sampled from a Gaussian distribution, to be used as inputs for image
          generation. Can be used to tweak the same generation with different
          prompts. If not provided, a latents tensor will ge generated by
          sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*): Pre-generated text
          embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
          weighting. If not provided, text embeddings will be generated from
          `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*): Pre-generated
          negative text embeddings. Can be used to easily tweak text inputs,
          *e.g.* prompt weighting. If not provided, negative_prompt_embeds will
          be generated from `negative_prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`): The output format
          of the generate image. Choose between
          [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or
          `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`): Whether or not to
          return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`]
          instead of a plain tuple.
        callback (`Callable`, *optional*): A function that will be called every
          `callback_steps` steps during inference. The function will be called
          with the following arguments:
          `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
        callback_steps (`int`, *optional*, defaults to 1): The frequency at
          which the `callback` function will be called. If not specified, the
          callback will be called at every step.
        cross_attention_kwargs (`dict`, *optional*): A kwargs dictionary that if
          specified is passed along to the `AttnProcessor` as defined under
          `self.processor` in
          [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        is_ddp (`bool`, *optional*, defaults to `False`): whether the unet is a
          `DistributedDataParallel` model. If `True`, the `height` and `width`
          arguments will be calculated using the unwrapped model.
        unet_copy (`torch.nn.Module`, *optional*, defaults to `None`): the
          pretrained model to calculate soft reward
        soft_reward (`bool`, *optional*, defaults to `False`): whether to use
          soft reward

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or
        `tuple`:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if
        `return_dict` is True, otherwise a `tuple.
        When returning a tuple, the first element is a list with the generated
        images, and the second element is a
        list of `bool`s denoting whether the corresponding generated image
        likely represents "not-safe-for-work"
        (nsfw) content, according to the `safety_checker`.
        latents_list (`List[torch.FloatTensor]`): A list of latents states
        unconditional_prompt_embeds (`List[torch.FloatTensor]`) A list of
        unconditional prompt embeddings
        guided_prompt_embeds (`List[torch.FloatTensor]`) A list of conditional
        prompt embeddings
        log_probs_list (`List[torch.FloatTensor]`): A list of log probabilities
        for each step
        kl_path_list (`List[torch.FloatTensor]`): A list of soft rewards for
        each step
    """
    # 0. Default height and width to unet
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
    def _decode(latents):
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type='pt', do_denormalize=do_denormalize)
        return image
    
    
    def _vila_single_score(single_input):
        input_batch = NestedMap(
          image=jnp.expand_dims(single_input, 0),
          ids=jnp.zeros((1,1,_MAX_TEXT_LEN), jnp.int32),
          paddings=jnp.zeros((1,1,_MAX_TEXT_LEN), jnp.int32),
        )
        ctx_p = base_layer.JaxContext.HParams(do_eval=True)
        with base_layer.JaxContext(ctx_p):
            preds = vila_reward_net.apply(
                {'params': vila_reward_net_states.mdl_vars['params']},
                input_batch,
                method=vila_reward_net.compute_predictions,
            )
        return jnp.squeeze(preds['quality_scores'])

    _vila_grad_fn = jax.jit(
        jax.vmap(jax.grad(_vila_single_score))  # now grad expects exactly one arg
    )


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

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
      batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
      batch_size = len(prompt)
    else:
      batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of
    # equation (2) of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
    # `guidance_scale = 1` corresponds to doing no classifier free guidance.
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

    kl_coeff = torch.tensor(kl_coeff, device=device).to(torch.float32)
    lookforward_fn = lambda r: r / kl_coeff

    # 7. Denoising loop
    num_warmup_steps = (
        len(timesteps) - num_inference_steps * self.scheduler.order
    )
    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for i, t in enumerate(timesteps):
        if is_tempering:
            lambda_t = (1 + gamma) ** i - 1
            if lambda_t>1:
                lambda_t=1
        else:
            lambda_t = 1
        
            
        prev_timestep = (t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps)
        prev_timestep = torch.clamp(prev_timestep, 0, self.scheduler.config.num_train_timesteps - 1)
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

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

        if do_guidance>0:
            with torch.enable_grad():
                tmp_noise_pred = noise_pred.detach().to(torch.float32).requires_grad_(True) 
                tmp_latents = latents.detach().to(torch.float32).requires_grad_(True) 

                tmp_pred_original_sample, _ = ddim_prediction_with_logprob(self.scheduler, tmp_noise_pred, t, tmp_latents,)
                if reward_fn_name == 'multi':
                    approx_guidance = 0
                    if ir_weight > 0:
                        tmp_ir_rewards = ir_reward_fn(_decode(tmp_pred_original_sample), prompt).to(torch.float32)
                        weighted_reward = lookforward_fn(tmp_ir_rewards).to(torch.float32)
                        ir_approx_guidance = torch.autograd.grad(
                            outputs=weighted_reward,
                            inputs=tmp_latents,
                            grad_outputs=torch.ones_like(weighted_reward),
                            retain_graph=True
                        )[0].detach().clone()
                        if grad_norm:
                            ir_approx_guidance = F.normalize(ir_approx_guidance, dim=1)

                        approx_guidance += ir_weight * ir_approx_guidance

                    if ir_weight < 1:
                        image0 = _decode(tmp_pred_original_sample)
                        resized_tensor = F.interpolate(image0, size=(224, 224), mode='bilinear', align_corners=False)
                        resized_hwc = resized_tensor.permute(0, 2, 3, 1)
                        jax_input = jnp.array(resized_hwc.detach().cpu().numpy())
                        grads = _vila_grad_fn(jax_input)
                        vila_guidance_image = torch.from_numpy(np.array(grads)).permute(0, 3, 1, 2).to(latents.device, latents.dtype)
                        vila_guidance_image = F.interpolate(vila_guidance_image, size=image0.shape[-2:], mode='bilinear', align_corners=False)
                        image = image0.to(torch.float32)
                        vila_approx_guidance = torch.autograd.grad(
                            outputs=image,
                            inputs=tmp_latents,
                            grad_outputs=vila_guidance_image,
                            retain_graph=True
                        )[0].detach()
                        vila_approx_guidance = vila_approx_guidance / kl_coeff
                        if grad_norm:
                            vila_approx_guidance = F.normalize(vila_approx_guidance, dim=1)

                        approx_guidance += (1 - ir_weight) * vila_approx_guidance
                else:
                    tmp_rewards = reward_fn(_decode(tmp_pred_original_sample) , prompt).to(torch.float32)
                    weighted_reward = lookforward_fn(tmp_rewards).to(torch.float32)
                    approx_guidance = (torch.autograd.grad(outputs=weighted_reward, inputs=tmp_latents,
                                                           grad_outputs=torch.ones_like(weighted_reward))[0].detach()).clone()

                
        prev_sample, prev_sample_mean = ddim_step_with_mean(self.scheduler, noise_pred, t, latents, **extra_step_kwargs)
        variance = get_variance(self.scheduler, t, prev_timestep)
        variance = eta**2 * _left_broadcast(variance, prev_sample.shape).to(device) 
        if do_guidance>0:
            prop_latents =  prev_sample + lambda_t * variance * approx_guidance
            latents = prop_latents.detach()
        else:
            latents =  prev_sample.detach() 
        
          
        latents = latents.to(prompt_embeds.dtype)
        latents_list.append(latents.detach().clone().cpu())

        if i == len(timesteps) - 1 or (
            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
        ):
          progress_bar.update()
          if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

    if output_type == "latent":
      image = latents
    elif output_type == "pil":
      latents = latents.detach()
      latents = latents.to(prompt_embeds.dtype)
      image_tensor = _decode(latents).detach().cpu()
      image_numpy = image_tensor.permute(0, 2, 3, 1).float().numpy()
      image = self.numpy_to_pil(image_numpy)
    else:
      image = _decode(latents)

    if (
        hasattr(self, "final_offload_hook")
        and self.final_offload_hook is not None
    ):
      self.final_offload_hook.offload()

    unconditional_prompt_embeds, guided_prompt_embeds = prompt_embeds.chunk(2)


    return image