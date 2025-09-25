"""
Simple demo showing StableDiffusionPipelineLoRArandom usage with multiple UNets
"""

from pipeline_stable_diffusion_LoRArandom import StableDiffusionPipelineLoRArandom
from diffusers import DDIMScheduler, UNet2DConditionModel
import torch

# Load main pipeline
pipe = StableDiffusionPipelineLoRArandom.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float32
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# Load additional UNet
unet2 = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    subfolder="unet"
).to("cuda")

# Load LoRA weights for each UNet
pipe.unet.load_attn_procs("path/to/lora_weights_1.bin")
unet2.load_attn_procs("path/to/lora_weights_2.bin")

# Generate image with multiple objectives
prompt = "a beautiful landscape"
mix_rate = 0.5

image = pipe.forward_ddim_multi_obj(
    prompt=prompt,
    unet_obj2=unet2,
    eta=1.0,
    mix_rate=mix_rate
).images[0]

# Save result
image.save("generated_image.png")
print("Image saved as generated_image.png")
