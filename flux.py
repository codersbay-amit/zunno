import torch
from diffusers.utils import check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from rembg import remove
from PIL import Image
import io

# Ensure the required version is installed
check_min_version("0.30.2")

def generate_mask_from_image(input_image, size=(768, 768)):
    """Generate a mask using rembg for the input image."""
    #input_image_data = input_image.tobytes()  # Convert PIL image to byte data
    mask_data = remove(input_image)  # Generate mask using rembg
    #mask_image = Image.open(io.BytesIO(mask_data)).convert("RGB")
    return mask_data.resize(size)

def build_pipeline():
    """Build and return the inpainting pipeline."""
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    return pipe

def inpaint_image(pipe, input_image, mask_image, prompt, size=(768, 768)):
    """Perform image inpainting based on the input image and mask."""
    generator = torch.Generator(device="cuda").manual_seed(24)
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=input_image,
        control_mask=mask_image,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=3.5
    ).images[0]
    return result

def outpaint(input_image, prompt,neg_prmpt='', size=(768, 768)):
    """Generate inpainted image from the given input image and prompt."""
    # Generate mask from the input image
    mask_image = generate_mask_from_image(input_image, size)

    # Build pipeline
    pipe = build_pipeline()

    # Inpaint the image
    result = inpaint_image(pipe, input_image, mask_image, prompt, size)

    # Save and display the result
    result.save("flux_inpaint.png")
    print("Successfully inpainted the image.")
    return result


