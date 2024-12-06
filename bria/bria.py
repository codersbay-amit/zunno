import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from bria.replace_bg.model.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from bria.replace_bg.model.controlnet import ControlNetModel
from bria.replace_bg.utilities import resize_image, remove_bg_from_image, paste_fg_over_image, get_control_image_tensor

def resize_and_prepare_image(image, size=(1024, 1024)):
    """Resize the input PIL image."""
    return resize_image(image)

def generate_mask_from_pil_image(image):
    """Generate a mask from the input PIL image."""
    # Save the PIL image to a temporary file to process it with remove_bg_from_image
    return remove_bg_from_image(image)

def prepare_control_tensor(pipe, image, mask):
    """Prepare the control tensor for image processing."""
    return get_control_image_tensor(pipe.vae, image, mask)

def build_pipeline():
    """Build and return the StableDiffusionXL ControlNet pipeline."""
    controlnet = ControlNetModel.from_pretrained("briaai/BRIA-2.3-ControlNet-BG-Gen", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16, vae=vae
    ).to('cuda:0')

    pipe.scheduler = EulerAncestralDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        steps_offset=1
    )

    return pipe

def generate_inpainted_image(pipe, control_tensor, prompt, negative_prompt, generator, num_steps=50, conditioning_scale=1.0):
    """Generate an inpainted image using the pipeline."""
    return pipe(
        negative_prompt=negative_prompt, 
        prompt=prompt,     
        controlnet_conditioning_scale=conditioning_scale, 
        num_inference_steps=num_steps,
        image=control_tensor,
        generator=generator
    ).images[0]

def outpaint(input_image, prompt, negative_prompt, size=(1024, 1024)):
    """Complete process for loading image, generating mask, and performing inpainting."""
    image = resize_and_prepare_image(input_image, size)

    # Step 2: Generate mask (remove background)
    mask = generate_mask_from_pil_image(input_image)

    # Step 3: Build the pipeline
    pipe = build_pipeline()

    # Step 4: Prepare control tensor
    control_tensor = prepare_control_tensor(pipe, image, mask)

    # Step 5: Set up the generator for reproducibility
    generator = torch.Generator(device="cuda:0").manual_seed(0)

    # Step 6: Generate inpainted image with reduced steps
    gen_img = generate_inpainted_image(pipe, control_tensor, prompt, negative_prompt, generator)

    # Step 7: Combine the generated image with the original foreground
    result_image = paste_fg_over_image(gen_img, image, mask)

    # Clear CUDA cache to free memory after generation
    torch.cuda.empty_cache()
