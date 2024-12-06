import spaces
import gradio as gr
import torch
from diffusers import (
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image
from replace_bg.model.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
from replace_bg.model.controlnet import ControlNetModel
from replace_bg.utilities import resize_image, remove_bg_from_image, paste_fg_over_image, get_control_image_tensor

controlnet = ControlNetModel.from_pretrained("briaai/BRIA-2.3-ControlNet-BG-Gen", torch_dtype=torch.float16) 
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("briaai/BRIA-2.3", controlnet=controlnet, torch_dtype=torch.float16, vae=vae).to('cuda:0')
pipe.scheduler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    steps_offset=1
)


@spaces.GPU
def generate_(prompt, negative_prompt, control_tensor, num_steps, controlnet_conditioning_scale, seed):
    generator = torch.Generator("cuda").manual_seed(seed)    
    gen_img = pipe(
        negative_prompt=negative_prompt, 
        prompt=prompt,     
        controlnet_conditioning_scale=float(controlnet_conditioning_scale), 
        num_inference_steps=num_steps,
        image = control_tensor,
        generator=generator
    ).images[0]
    
    return gen_img

@spaces.GPU
def process(input_image, prompt, negative_prompt, num_steps, controlnet_conditioning_scale, seed):
    
    image = resize_image(input_image)
    mask = remove_bg_from_image(image)
    control_tensor = get_control_image_tensor(pipe.vae, image, mask)    
  
    gen_image = generate_(prompt, negative_prompt, control_tensor, num_steps, controlnet_conditioning_scale, seed)
    result_image = paste_fg_over_image(gen_image, image, mask)

    return result_image



block = gr.Blocks().queue()

with block:
    gr.Markdown("## BRIA Background Generation")
    gr.HTML('''
      <p style="margin-bottom: 10px; font-size: 94%">
        This is a demo for ControlNet background generation that using BRIA 2.3 text-to-image model as backbone.
        Trained on licensed data, BRIA 2.3 provide full legal liability coverage for copyright and privacy infringement.
        Go <a href="https://huggingface.co/briaai/BRIA-2.3-ControlNet-BG-Gen" target="_blank"> here</a> for the BRIA 2.3 ControlNet Background Generation model card or Contact <a href="https://bria.ai/contact-us/"> Bria</a> for more information.
      </p>
    ''')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="pil", label="Upload", elem_id="image_upload", height=600) # None for upload, ctrl+v and webcam
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative prompt", value="Logo,Watermark,Text,Ugly,Morbid,Extra fingers,Poorly drawn hands,Mutation,Blurry,Extra limbs,Gross proportions,Missing arms,Mutated hands,Long neck,Duplicate,Mutilated,Mutilated hands,Poorly drawn face,Deformed,Bad anatomy,Cloned face,Malformed limbs,Missing legs,Too many fingers")
            num_steps = gr.Slider(label="Number of steps", minimum=10, maximum=100, value=30, step=1)
            controlnet_conditioning_scale = gr.Slider(label="ControlNet conditioning scale", minimum=0.1, maximum=2.0, value=1.0, step=0.05)
            seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True,)
            run_button = gr.Button(value="Generate")
            
            
        with gr.Column():
            result_gallery = gr.Image(label='Output', type="pil", show_label=True, elem_id="output-img") 
            # result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=[1], height=600)
    ips = [input_image, prompt, negative_prompt, num_steps, controlnet_conditioning_scale, seed]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

    gr.Examples(
                examples=[
                    ["./example1.png"],
                    ["./example2.png"],
                    ["./example3.png"],
                    ["./example4.png"],
                ],
                fn=process,
                inputs=[input_image],
                cache_examples=False,
    )


block.launch(debug = True)