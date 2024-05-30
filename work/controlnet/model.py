from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DDPMScheduler
)
import torch

controlnet = ControlNetModel.from_pretrained(
    "alimama-creative/EcomXL_controlnet_inpaint",
    use_safetensors=True,
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
