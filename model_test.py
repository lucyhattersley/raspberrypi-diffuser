from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import time
 
start_time = time.time() # so we can time runtime
models_dir = "/home/lucy/Models/" # store our models here

# Name of model
# model_name = "mitsua-diffusion-one"  # Pick a  model from the Models folder

# Stable diffusion v1.5
model_name = "stable-diffusion-v1-5" # Pick a  model from the Models folder

model = models_dir + model_name
pipe = StableDiffusionPipeline.from_pretrained(model, low_cpu_mem_usage=True) # Add torch_dtype=torch.float32 for Mit   
pipe = pipe.to("cpu")

prompt = "a cute magical flying space dog with a cape, fantasy space  art drawn by concept artists, golden colour, high quality, highly detailed, elegant, sharp focus, concept art, character concepts, digital painting, mystery, adventure"
image = pipe(prompt, num_inference_steps=35, width=512, height=512).images[0]

elapsed_time = time.time() - start_time
mins, secs = int(elapsed_time // 60), elapsed_time % 60
elapsed_str = f"{mins} min {secs:.2f} sec"
filename = f"space-dog{model_name}_{elapsed_str.replace(' ', '_')}.png"
image.save(filename)
