from rembg import remove
import random
import torch
from controlnet_aux import ZoeDetector
from PIL import Image, ImageOps
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)

def scale_and_paste(original_image):
    aspect_ratio = original_image.width / original_image.height

    if original_image.width > original_image.height:
        new_width = 1024
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = 1024
        new_width = round(new_height * aspect_ratio)

    resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
    resized_original = resized_original.convert('RGBA')
    white_background = Image.new("RGBA", (1024, 1024), "white")
    x = (1024 - new_width) // 2
    y = (1024 - new_height) // 2
    white_background.paste(resized_original, (x, y), resized_original)

    return resized_original, white_background

def outpaint(image, prompt, negative_prompt, seed: int = None):
    resized_img, white_bg_image = scale_and_paste(image)
    
    # Load the depth detection model for ControlNet (Zoe)
    zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
    image_zoe = zoe(white_bg_image, detect_resolution=512, image_resolution=1024)

    # Load ControlNet models with FP16 precision to reduce memory usage
    controlnets = [
        ControlNetModel.from_pretrained(
            "destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
        ),
        ControlNetModel.from_pretrained(
            "diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16
        ),
    ]

    # Use the FP16 variant of the VAE model to save memory
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

    # Load pipeline with models and VAE
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        controlnet=controlnets, 
        vae=vae
    ).to("cuda")

    # Set a random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Use a random generator to ensure consistency
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Reduce memory usage: 
    # - Use lower `guidance_scale` to reduce memory load
    # - Reduce inference steps for faster, less memory-intensive inference
    image = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        image=[resized_img, image_zoe],
        guidance_scale=9.0,  # Decrease this value if necessary
        num_inference_steps=50,  # Lower inference steps to reduce memory usage
        generator=generator,
        controlnet_conditioning_scale=[0.5, 0.8],
        control_guidance_end=[0.9, 0.6],
    ).images[0]

    # Clear the pipeline models to free up GPU memory
    del pipeline
    torch.cuda.empty_cache()

    x = (1024 - resized_img.width) // 2
    y = (1024 - resized_img.height) // 2
    resized_img=remove(resized_img).convert('RGBA')
    #image.paste(resized_img, (x, y), resized_img)
    return image
