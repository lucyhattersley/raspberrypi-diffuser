import os # We need this to expand ~ to your actual home dir
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# The expanded path to the model
model = os.path.expanduser("~/Models/stable-diffusion-xl")

pipe = StableDiffusionXLPipeline.from_pretrained(model, low_cpu_mem_usage=True) # Load the model
pipe = pipe.to("cpu") # Move the model to the CPU

prompt = "an astronaut riding a horse on mars" # Set the prompt
image = pipe(prompt, num_inference_steps=35, width=1024, height=1024).images[0]

image.save("astro-horse-sdxl-35-steps.png") 
