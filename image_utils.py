import requests
from rembg.bg import remove
from PIL import Image

# Define the API endpoint
url = "http://127.0.0.1:5000/generate_outpaint"

# Image and mask file paths
mask_path = 'mask.jpg'
image_path = "shoes.jpg"  # Path to the image file

# Generate the mask using rembg
mask = remove(Image.open(image_path), only_mask=True)
mask.save(mask_path)

# JSON-like data for the prompts (sent as form data)
data = {
    "prompt": "A futuristic cityscape with a sunset",
}

# Open the image and mask files in binary mode
files = {
    'image': open(image_path, 'rb'),
}

# Send the POST request with the data and files
response = requests.post(url, data=data, files=files)

# Handle the response
if response.status_code == 200:
    # Save the outpainted image to a file
    with open("outpainted_image.png", "wb") as f:
        f.write(response.content)
    print("Outpainted image saved as 'outpainted_image.png'")
else:
    print(f"Error: {response.status_code}")
    print(response.json())  # Print error details if the request failed