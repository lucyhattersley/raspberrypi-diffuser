from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import time
 
start_time = time.time() # so we can time runtime
models_dir = "/home/lucy/Models/" # store our models here

# Mitsua Stable Diffusion requres we specifiy float32
# pipe = StableDiffusionPipeline.from_pretrained("/home/lucy/Models/mitsua-diffusion-one", torch_dtype=torch.float32, low_cpu_mem_usage=True)

# Stable diffusion v1.5
model_name = "stable-diffusion-v1-5" # Pick a  model from the Models folder
model = models_dir + model_name
pipe = StableDiffusionPipeline.from_pretrained(model, low_cpu_mem_usage=True)
pipe = pipe.to("cpu")

prompt = "a photograph of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=35, width=512, height=512).images[0]

elapsed_time = time.time() - start
filename = f"astro-horse-mars_{model}_{elapsed_time:.2f}s.png"
image.save(filename)
