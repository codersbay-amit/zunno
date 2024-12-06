from face_predict import Predictor

predicter=Predictor()
predicter.setup()
from PIL import ImageOps
def add_padding(img):
    # Open the image
    
    
    # Get the original width and height of the image
    width, height = img.size
    
    # Calculate padding for each side to make the image square
    if width == height:
        print("The image is already square.")
        return img
    
    # Determine the size of the new square image (max of width or height)
    new_size = max(width, height)
    
    # Calculate padding (in pixels) to add on each side
    padding_left = (new_size - width) // 2
    padding_top = (new_size - height) // 2
    padding_right = new_size - width - padding_left
    padding_bottom = new_size - height - padding_top
    
    # Add padding to the image to make it square with white background
    square_img = ImageOps.expand(img, (padding_left, padding_top, padding_right, padding_bottom), fill=(255, 255, 255))  # White padding
    return square_img

def swap(source,target):
   source.save('source.jpg')
   target.save('target.jpg')
   output=predicter.predict('source.jpg','target.jpg')
   return add_padding(output)    

