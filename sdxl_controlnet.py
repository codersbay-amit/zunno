from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

class ControlNetPipeline:
    def __init__(self, 
                 controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0", 
                 vae_model: str = "madebyollin/sdxl-vae-fp16-fix",
                 stable_diffusion_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 controlnet_conditioning_scale: float = 0.5, 
                 torch_dtype: torch.dtype = torch.float16):
        """
        Initialize the pipeline with models and parameters.
        
        Parameters:
            controlnet_model (str): Pretrained ControlNet model identifier.
            vae_model (str): Pretrained VAE model identifier.
            stable_diffusion_model (str): Pretrained Stable Diffusion model identifier.
            controlnet_conditioning_scale (float): The scale factor for controlnet conditioning.
            torch_dtype (torch.dtype): The dtype used for the model (e.g., torch.float16).
        """
        # Load models
        self.controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch_dtype)
        self.vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch_dtype)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            stable_diffusion_model,
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=torch_dtype,
        )
        self.pipe.enable_model_cpu_offload()

        # Set conditioning scale
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

    def generate_image(self, prompt: str, negative_prompt: str, image: Image) -> Image:
        """
        Generate an image using the ControlNet and Stable Diffusion pipeline with the provided prompt and image input.
        
        Parameters:
            prompt (str): The positive text prompt.
            negative_prompt (str): The negative text prompt to avoid unwanted elements.
            image (Image): The preprocessed control image (e.g., edge map in PIL format).
        
        Returns:
            Image: The generated image.
        """
        # Run the pipeline
        generated_images = self.pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            image=image, 
            controlnet_conditioning_scale=self.controlnet_conditioning_scale
        ).images
        torch.cuda.empty_cache()        
        # Return the generated image
        return generated_images[0]

