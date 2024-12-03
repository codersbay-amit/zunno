import ollama
def prompt():
    return ollama.chat(
	model="llava",
	messages=[
		{
			'role': 'user',
			'content': '''Please analyze the following image and write a prompt to regenerate a similar background with the key features intact
                              Provide a short explanation (less than 60 words) about the elements in the image,
                              such as colors, textures, and composition, that should be preserved in the generated background
			      note: explain under 60 words only
                        ''',
			'images': ['image.png']
		}
	]
)['message']['content']


