import ollama

def summarize_prompt(prmpt):
    prmpt= ollama.chat(
        model="llava",
        messages=[
                {
                        'role': f'user',
                        'content': '''
                         please rewrite this prompt into 70 words.  text is '{prmpt}'
                        ''',
                       
                }
        ]
)['message']['content']
    print(prmpt)
    return prmpt
def prompt(image_path):
    prmpt= ollama.chat(
	model="llava",
	messages=[
		{
			'role': 'user',
			'content': '''Please analyze the following image and write a prompt to regenerate a similar background with the key features intact
                              Provide a short explanation (less than 60 words) about the elements in the image,
                              such as colors and colorscheme, textures, and composition,and orientaion that should be preserved in the generated background
			      note: explain under 60 words only
                        output_format:Minimalist interior with a light blue textured wall,
                                     soft natural sunlight casting diagonal shadows through an 
                                     unseen window. The wooden floor has a light, natural grain finish. 
                                     A small white ceramic pot containing a green indoor plant with
                        ''',
			'images': [image_path]
		}
	]
)['message']['content']
    return prmpt
