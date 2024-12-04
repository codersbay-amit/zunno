import requests
from rembg.bg import remove
from PIL import Image
from bria.bria import outpaint
#from flux import outpaint
#from canny import outpaint
from face_swapper import swap
from llava import prompt
#from new_bg_generator import BackgroundGenerator
from transformers import pipeline
def rem_bg(image):
#image_path = "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg"
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    #pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask    
    pillow_image = pipe(image) # applies mask on input and returns a pillow image
    return pillow_image
def apparel(source,target,background):
    output=swap(source,target)
    background.save('bg.jpg')
    neg_prmpt="""Low quality, blurry, Do not include any distracting patterns, heavy textures, 
                      bright colors, or elements that clash with the product. Avoid busy designs, gradients, or any elements that make the
                      background look overly complex or unprofessional. The extended background
                      should remain clean, neutral, and simple, maintaining focus on the product."""
    #return output
    output=rem_bg(output)
    #return output
    prmpt=prompt('bg.jpg')
    white_image=Image.new(mode='RGB',size=output.size,color=(255,255,255))
    white_image.paste(output,(0,0),output)
    print(white_image.size)
    #g=BackgroundGenerator()
    fo=outpaint(white_image,prmpt,neg_prmpt)
    #fo=white_image
    print(fo.size)
    fo=fo.resize(output.size)
    print(fo.size,white_image.size)
    fo.paste(output,(0,0),output)
    return fo
    

