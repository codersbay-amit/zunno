from rembg.bg import remove
from PIL import Image

image=Image.open('shoes.jpg')
image=remove(image)
image.show()