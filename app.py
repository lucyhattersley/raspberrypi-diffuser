import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionPipeline
from PIL import Image

# The expanded path to the model
model = os.path.expanduser("~/Models/stable-diffusion-v1-5")

pipe = StableDiffusionPipeline.from_pretrained(model, low_cpu_mem_usage=True) # Load the model
pipe = pipe.to("cpu") # Move the model to the CPU

prompt = "an astronaut riding a horse on mars" # Set the prompt
image = pipe(prompt, num_inference_steps=5, width=512, height=512).images[0]

image.save("astro-horse.png") 
