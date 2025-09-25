"""Extension of diffusers.StableDiffusionPipeline."""

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

class StableDiffusionPipelineLoRArandom(StableDiffusionPipeline):
    """Extension of diffusers.StableDiffusionPipeline."""

    @torch.no_grad()
    def forward_ddim_multi_obj(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        unet_obj2 = None,  # single unet model or list of unet models
        mix_rate: Union[float, List[float]] = 0.5,  # prob of self.unet or probability vector
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        
        This version supports probabilistic selection among multiple UNet models during the denoising process.
        
        Args:
            unet_obj2: Single UNet model or list of UNet models to mix with self.unet
            mix_rate: If unet_obj2 is a single model, this is the probability of using self.unet (float).
                     If unet_obj2 is a list, this should be a probability vector of length len(unet_obj2)+1,
                     where mix_rate[0] is the probability of self.unet, and mix_rate[i+1] is the probability
                     of unet_obj2[i].
        """
        
        # Validate and process unet_obj2 and mix_rate parameters
        if unet_obj2 is None:
            # If no additional models, use only self.unet
            all_unets = [self.unet]
            mix_probs = [1.0]
        else:
            # Handle both single model and list of models
            if not isinstance(unet_obj2, list):
                unet_obj2 = [unet_obj2]
            
            # Create the complete list of UNets: [self.unet, *unet_obj2]
            all_unets = [self.unet] + unet_obj2
            
            # Process mix_rate
            if isinstance(mix_rate, (int, float)):
                # Original behavior: mix_rate is probability of self.unet
                if len(unet_obj2) == 1:
                    mix_probs = [float(mix_rate), 1.0 - float(mix_rate)]
                else:
                    raise ValueError(
                        f"When unet_obj2 contains {len(unet_obj2)} models, mix_rate must be a list of "
                        f"{len(unet_obj2) + 1} probabilities, not a single float."
                    )
            else:
                # mix_rate should be a probability vector
                if len(mix_rate) != len(all_unets):
                    raise ValueError(
                        f"mix_rate length ({len(mix_rate)}) must equal the number of UNet models "
                        f"({len(all_unets)} = 1 + {len(unet_obj2)})"
                    )
                
                mix_probs = list(mix_rate)
                
                # Validate probabilities are non-negative
                if any(p < 0 for p in mix_probs):
                    raise ValueError("All probabilities in mix_rate must be non-negative")
                
                # Normalize probabilities to sum to 1
                total_prob = sum(mix_probs)
                if total_prob <= 0:
                    raise ValueError("Sum of probabilities in mix_rate must be positive")
                mix_probs = [p / total_prob for p in mix_probs]
        
        # Convert to cumulative probabilities for efficient sampling
        cumulative_probs = []
        cumsum = 0.0
        for prob in mix_probs:
            cumsum += prob
            cumulative_probs.append(cumsum)
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
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

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop with probabilistic UNet selection
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Probabilistically select UNet based on probability vector
                rand_val = torch.rand(1).item()
                selected_idx = 0
                for idx, cum_prob in enumerate(cumulative_probs):
                    if rand_val <= cum_prob:
                        selected_idx = idx
                        break
                
                current_unet = all_unets[selected_idx]

                # predict the noise residual using the selected UNet
                noise_pred = current_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)