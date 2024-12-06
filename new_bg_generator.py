import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageOps
from transparent_background import Remover



class BackgroundGenerator:
    def __init__(self, model_id="yahoo-inc/photo-background-generation", device="cuda"):
        self.device = device
        self.pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id).to(device)
        self.remover = Remover(mode='base')  # Initialize background remover with base mode

    @staticmethod
    def resize_with_padding(img, expected_size):
        """Resizes and pads the image to the expected size."""
        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return ImageOps.expand(img, padding)

    def clear_cuda_cache(self):
        """Clears CUDA cache to prevent memory issues."""
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def generate_image(self, input_image, prompt, seed=122220, cond_scale=1.5):
        """
        Generates an image based on the given prompt and input PIL image.
        
        :param input_image: PIL image as input.
        :param prompt: Text prompt for image generation.
        :param seed: Seed for reproducibility.
        :param cond_scale: Conditioning scale for the pipeline.
        :return: Generated image.
        """
        try:
            # Resize and pad the input image
            img = self.resize_with_padding(input_image, (512, 512))
            
            # Generate foreground mask
            fg_mask = self.remover.process(img, type='map')
            mask = ImageOps.invert(fg_mask)
            
            # Set up generator for reproducibility
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Run the pipeline
            with torch.autocast(self.device), torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    image=img,
                    mask_image=mask,
                    control_image=mask,
                    num_images_per_prompt=1,
                    generator=generator,
                    num_inference_steps=200,
                    guess_mode=False,
                    controlnet_conditioning_scale=cond_scale
                )
                return result.images[0]
        except torch.cuda.OutOfMemoryError:
            self.clear_cuda_cache()
            print("CUDA out of memory. Retrying with reduced settings...")
            try:
                with torch.autocast(self.device), torch.no_grad():
                    # Retry with fewer inference steps and reduced memory usage
                    result = self.pipeline(
                        prompt=prompt,
                        image=img,
                        mask_image=mask,
                        control_image=mask,
                        num_images_per_prompt=1,
                        generator=generator,
                        num_inference_steps=50,  # Reduced steps
                        guess_mode=False,
                        controlnet_conditioning_scale=cond_scale
                    )
                    return result.images[0]
            except torch.cuda.OutOfMemoryError:
                print("Failed again due to insufficient GPU memory. Consider reducing image size or using CPU.")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
