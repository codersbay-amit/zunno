import os
from flask import Flask, request, jsonify, send_file
from PIL import Image,ImageOps
from io import BytesIO
from canny import generate_outpaint
from rembg.bg import remove

# Initialize Flask app
app = Flask(__name__)

# Helper function to convert image to BytesIO (for Flask response)
def image_to_bytes(image: Image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

# Flask route to handle image generation requests
@app.route("/generate_outpaint", methods=["POST"])
def generate_outpaint_route():
    try:
        # Get the input data from the request
        data = request

        prompt = data.form.get('prompt')
        negative_prompt =   "Low quality, blurry",
        

        # Get image and mask from the request (they are expected to be files)
        image_file = request.files['image']
       

        # Open the image and mask
        image = Image.open(image_file.stream).convert("RGB")
        mask = ImageOps.invert(remove(image,only_mask=True) ) # Assuming the mask is grayscale
        mask.show()
        # Generate the outpainted image
        outpainted_image = generate_outpaint(prompt, negative_prompt, image, mask)

        # Convert the result image to a byte stream
        img_byte_arr = image_to_bytes(outpainted_image)

        # Return the image as a response
        return send_file(img_byte_arr, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
