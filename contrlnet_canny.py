import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from diffusers.utils import load_image

def canny_edge_detection(input_image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """
    Converts input image to a Canny edge detection image.
    
    Args:
        input_image (Image.Image): The input PIL image to apply edge detection.
        low_threshold (int): Low threshold for Canny edge detection.
        high_threshold (int): High threshold for Canny edge detection.
        
    Returns:
        Image.Image: PIL image after applying Canny edge detection.
    """
    # Convert PIL image to numpy array
    image = np.array(input_image)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)
    
    # Convert single channel to three-channel image for consistency
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    # Convert back to PIL Image
    return Image.fromarray(edges)

def load_controlnet_model(model_name: str = "lllyasviel/sd-controlnet-canny") -> ControlNetModel:
    """
    Loads the ControlNet model from HuggingFace repository.
    
    Args:
        model_name (str): The name of the pretrained model to load.
        
    Returns:
        ControlNetModel: The loaded ControlNet model.
    """
    return ControlNetModel.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

def setup_diffusion_pipeline(controlnet: ControlNetModel) -> StableDiffusionControlNetPipeline:
    """
    Sets up the StableDiffusionControlNetPipeline with the provided controlnet model.
    
    Args:
        controlnet (ControlNetModel): The ControlNet model to use in the pipeline.
        
    Returns:
        StableDiffusionControlNetPipeline: The configured diffusion pipeline.
    """
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    
    # Set the scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Enable memory-efficient attention if xformers is available
    pipe.enable_xformers_memory_efficient_attention()
    
    # Enable model CPU offload
    pipe.enable_model_cpu_offload()
    
    return pipe

def generate_image_from_prompt_and_edges(pipe: StableDiffusionControlNetPipeline, prompt: str, edge_image: Image.Image, num_inference_steps: int = 20) -> Image.Image:
    """
    Generates an image based on a prompt and edge-detected image using the StableDiffusionControlNetPipeline.
    
    Args:
        pipe (StableDiffusionControlNetPipeline): The diffusion pipeline to generate images.
        prompt (str): The text prompt for image generation.
        edge_image (Image.Image): The edge-detected input image.
        num_inference_steps (int): The number of inference steps for image generation.
        
    Returns:
        Image.Image: The generated image.
    """
    return pipe(prompt, edge_image, num_inference_steps=num_inference_steps).images[0]

def outpaint(input_image: Image.Image,prompt: str,neg_prmpt='') -> Image.Image:
    """
    Main method to process an image (apply Canny edge detection) and generate a new image based on a prompt.
    
    Args:
        prompt (str): The text prompt for image generation.
        input_image (Image.Image): The input image to apply edge detection.
        
    Returns:
        Image.Image: The generated image.
    """
    # Step 1: Apply Canny edge detection
    edge_image = canny_edge_detection(input_image)
    
    # Step 2: Load the ControlNet model
    controlnet = load_controlnet_model()
    
    # Step 3: Set up the diffusion pipeline
    pipe = setup_diffusion_pipeline(controlnet)
    
    # Step 4: Generate an image based on the prompt and edge-detected image
    generated_image = generate_image_from_prompt_and_edges(pipe, prompt, edge_image)
    
    return generated_image
