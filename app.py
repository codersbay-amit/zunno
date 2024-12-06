import os
import requests
import base64
from flask import Flask, request, jsonify, send_file
from PIL import Image,ImageOps
from io import BytesIO
from canny import outpaint
import io
from rembg.bg import remove
from llava import prompt
from image_utils import apparel
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

def image_to_base64(image: Image) -> str:
    """
    Converts a PIL Image to a Base64-encoded string.
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # Change format to 'JPEG' if necessary
    img_byte_arr.seek(0)  # Rewind the byte stream
    
    # Encode the image bytes to Base64 and return as a string
    base64_string = base64.b64encode(img_byte_arr.read()).decode('utf-8')
    return base64_string
def load_image_from_url(url):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Ensure the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the image content into a byte stream
        img_data = BytesIO(response.content)
        
        # Open the image with PIL
        img = Image.open(img_data).convert('RGB')
        return img
    else:
        raise Exception(f"Failed to retrieve image from URL: {url}, Status Code: {response.status_code}")

# Helper function to convert image to BytesIO (for Flask response)
def image_to_bytes(image: Image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

from PIL import Image, ImageOps

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
    # Save the imag 
# Flask route to handle image generation requests
@app.route("/non_apparel", methods=["POST"])
def generate_outpaint_route():
    try:
        # Get the input data from the request
        data = request


        negative_prompt ="""Low quality, blurry, Do not include any distracting patterns, heavy textures, 
                      bright colors, or elements that clash with the product. Avoid busy designs, shadows, gradients, or any elements that make the
                      background look overly complex or unprofessional. The extended background
                      should remain clean, neutral, and simple, maintaining focus on the product."""


        # Get image and mask from the request (they are expected to be files)
        image_file_url = request.form['background_url']
        background_image_url = request.form['image_url']

        # Open the image and mask

        image = add_padding(Image.open(load_image_from_url(image_file_url)))
        #mask = ImageOps.invert(remove(image,only_mask=True) )
        #mask.show()
        # Generate the outpainted image
        background_image=load_image_from_url(image_file_url)
        background_image.save('image.png')
        prmpt=prompt('image.png')
        outpainted_image=outpaint(image,prmpt,negative_prompt)
        base64_string = image_to_base64(outpainted_image)
        return jsonify({'image_base64':base64_string})
        # Return the image as a JSON response with the Base64 string

    except Exception as e:
        # Handle any errors during processing
            return jsonify({"error": str(e)}), 500

@app.route("/apparel", methods=["POST"])
def generate_apparel():
    try:
        # Get the input data from the request
        data = request
        
        negative_prompt ="""Low quality, blurry, Do not include any distracting patterns, heavy textures, 
                      bright colors, or elements that clash with the product. Avoid busy designs, shadows, gradients, or any elements that make the
                      background look overly complex or unprofessional. The extended background
                      should remain clean, neutral, and simple, maintaining focus on the product."""
        

        # Get image and mask from the request (they are expected to be files)
        target_url = request.form['cloth_url']
        source_url = request.form['avatar_url']
        background_image_url = request.form['background_url']
        
        # Open the image and mask
        target=load_image_from_url(source_url)
        source=load_image_from_url(target_url)
        background=load_image_from_url(background_image_url)
        source=source.resize((1024,1024))
        print('all images read success')        
        result=apparel(source,target,background)
        base64_string = image_to_base64(result)
        return jsonify({'image_base64':base64_string}),200
    except Exception as e:
            print(e)
        # Handle any errors during processing
            return jsonify({"error": str(e)}), 500
# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
