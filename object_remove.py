from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
from PIL import Image

# Initialize the pipeline
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

def inpaint_image(image: Image.Image, mask: Image.Image, prompt: str = "erase", strength: float = 0.99, guidance_scale: float = 8.0, num_inference_steps: int = 20) -> Image.Image:
    """
    Inpaint an image based on a provided mask and prompt.

    Args:
    - image (PIL.Image.Image): The original image to be inpainted.
    - mask (PIL.Image.Image): The mask specifying the areas to inpaint.
    - prompt (str, optional): The prompt to guide inpainting (default is "erase").
    - strength (float, optional): The strength of the inpainting effect (default is 0.99).
    - guidance_scale (float, optional): The guidance scale for the model (default is 8.0).
    - num_inference_steps (int, optional): Number of steps for the inpainting process (default is 20).

    Returns:
    - PIL.Image.Image: The inpainted image.
    """
    # Resize the image and mask to 1024x1024 if not already
    image = image.resize((1024, 1024))
    mask = mask.resize((1024, 1024))

    # Convert the image and mask to the appropriate device and dtype
    generator = torch.Generator(device="cuda").manual_seed(0)

    # Perform inpainting using the pipeline
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
    )

    # Return the resulting inpainted image as a PIL image
    return result.images[0]
