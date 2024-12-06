
import random
import torch
import cv2
import numpy as np
from huggingface_hub import snapshot_download
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from diffusers.utils import load_image
from Kolors_Controlnet_and_IPA.kolors.pipelines.pipeline_controlnet_xl_kolors_img2img import StableDiffusionXLControlNetImg2ImgPipeline
from Kolors_Controlnet_and_IPA.kolors.models.modeling_chatglm import ChatGLMModel
from Kolors_Controlnet_and_IPA.kolors.models.tokenization_chatglm import ChatGLMTokenizer
from Kolors_Controlnet_and_IPA.kolors.models.controlnet import ControlNetModel
from diffusers import  AutoencoderKL
from Kolors_Controlnet_and_IPA.kolors.models.unet_2d_condition import UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from PIL import Image
from Kolors_Controlnet_and_IPA.annotator.midas import MidasDetector
from Kolors_Controlnet_and_IPA.annotator.dwpose import DWposeDetector
from Kolors_Controlnet_and_IPA.annotator.util import resize_image, HWC3


device = "cuda"
ckpt_dir = snapshot_download(repo_id="Kwai-Kolors/Kolors")
ckpt_dir_depth = snapshot_download(repo_id="Kwai-Kolors/Kolors-ControlNet-Depth")
ckpt_dir_canny = snapshot_download(repo_id="Kwai-Kolors/Kolors-ControlNet-Canny")
ckpt_dir_ipa = snapshot_download(repo_id="Kwai-Kolors/Kolors-IP-Adapter-Plus")
ckpt_dir_pose = snapshot_download(repo_id="Kwai-Kolors/Kolors-ControlNet-Pose")

text_encoder = ChatGLMModel.from_pretrained(f'{ckpt_dir}/text_encoder', torch_dtype=torch.float16).half().to(device)
tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half().to(device)
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half().to(device)

controlnet_depth = ControlNetModel.from_pretrained(f"{ckpt_dir_depth}", revision=None).half().to(device)
controlnet_canny = ControlNetModel.from_pretrained(f"{ckpt_dir_canny}", revision=None).half().to(device)
controlnet_pose = ControlNetModel.from_pretrained(f"{ckpt_dir_pose}", revision=None).half().to(device)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(f'{ckpt_dir_ipa}/image_encoder',  ignore_mismatched_sizes=True).to(dtype=torch.float16, device=device)
ip_img_size = 336
clip_image_processor = CLIPImageProcessor(size=ip_img_size, crop_size=ip_img_size )


pipe_canny = StableDiffusionXLControlNetImg2ImgPipeline(
    vae=vae,
    controlnet = controlnet_canny,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    image_encoder=image_encoder,
    feature_extractor=clip_image_processor,
    force_zeros_for_empty_prompt=False
)



for pipe in [pipe_depth]:
    if hasattr(pipe.unet, 'encoder_hid_proj'):
        pipe.unet.text_encoder_hid_proj = pipe.unet.encoder_hid_proj
        
pipe_depth.load_ip_adapter(f'{ckpt_dir_ipa}' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])
pipe_canny.load_ip_adapter(f'{ckpt_dir_ipa}' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])
pipe_pose.load_ip_adapter(f'{ckpt_dir_ipa}' , subfolder="", weight_name=["ip_adapter_plus_general.bin"])


def process_canny_condition(image, canny_threods=[100,200]):
    np_image = image.copy()
    np_image = cv2.Canny(np_image, canny_threods[0], canny_threods[1])
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    np_image = HWC3(np_image)
    return Image.fromarray(np_image)

model_midas = MidasDetector()


def process_depth_condition_midas(img, res = 1024):
    h,w,_ = img.shape
    img = resize_image(HWC3(img), res)
    result = HWC3(model_midas(img))
    result = cv2.resize(result, (w,h))
    return Image.fromarray(result)

model_dwpose = DWposeDetector()

def process_dwpose_condition(image, res=1024):
    h,w,_ = image.shape
    img = resize_image(HWC3(image), res)
    out_res, out_img = model_dwpose(image) 
    result = HWC3(out_img)
    result = cv2.resize( result, (w,h) )
    return Image.fromarray(result)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024



def infer_canny(prompt, 
          image = None, 
          ipa_img = None,
          negative_prompt = "nsfw，脸部阴影，低分辨率，糟糕的解剖结构、糟糕的手，缺失手指、质量最差、低质量、jpeg伪影、模糊、糟糕，黑脸，霓虹灯", 
          seed = 66, 
          randomize_seed = False,
          guidance_scale = 5.0, 
          num_inference_steps = 50,
          controlnet_conditioning_scale = 0.5,
          control_guidance_end = 0.9,
          strength = 1.0,
          ip_scale = 0.5,
        ):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    init_image = resize_image(image,  MAX_IMAGE_SIZE)
    pipe = pipe_canny.to("cuda")
    pipe.set_ip_adapter_scale([ip_scale])
    condi_img = process_canny_condition(np.array(init_image))
    image = pipe(
        prompt= prompt ,
        image = init_image,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        control_guidance_end = control_guidance_end, 
        ip_adapter_image=[ipa_img],
        strength= strength , 
        control_image = condi_img,
        negative_prompt= negative_prompt , 
        num_inference_steps= num_inference_steps, 
        guidance_scale= guidance_scale,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
    return [condi_img, image], seed

