import requests
from rembg.bg import remove
from PIL import Image
from canny import outpaint
from face_swapper import swap
from llava import prompt

def apparel(source,target,background):
    output=swap(source,target)
    output.save('bg.jpg')
    neg_prmpt="""Low quality, blurry, Do not include any distracting patterns, heavy textures, 
                      bright colors, or elements that clash with the product. Avoid busy designs, shadows, gradients, or any elements that make the
                      background look overly complex or unprofessional. The extended background
                      should remain clean, neutral, and simple, maintaining focus on the product."""
    prmpt=prompt('bg.jpg')
    return outpaint(output,prmpt,neg_prmpt)
