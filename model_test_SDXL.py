import os
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch
import time
 
start_time = time.time() # so we can time runtime
models_dir = os.path.expanduser("~/Models/")

# Stable diffusion v1.5
model_name = "stable-diffusion-xl-base-1.0" # Pick a  model from the Models folder

model = models_dir + model_name
pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float32, low_cpu_mem_usage=True) # Add torch_dtype=torch.float32 for Mitsua Diffusion One
pipe = pipe.to("cpu")

prompt = "an astronaut riding a horse"
image = pipe(prompt, num_inference_steps=35, width=1024, height=1024).images[0]

elapsed_time = time.time() - start_time
mins, secs = int(elapsed_time // 60), elapsed_time % 60
elapsed_str = f"{mins} min {secs:.2f} sec"
filename = f"astro-horse_{model_name}_{elapsed_str.replace(' ', '_')}.png"
image.save(filename)
