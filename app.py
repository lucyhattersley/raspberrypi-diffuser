from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

# Mitsua Stable Diffusion requres we specifiy float3
# pipe = StableDiffusionPipeline.from_pretrained("/home/lucy/Models/mitsua-diffusion-one", torch_dtype=torch.float32, low_cpu_mem_usage=True)

# Stable diffusion v1.5
pipe = StableDiffusionPipeline.from_pretrained("/home/lucy/Models/stable-diffusion-v1-5", low_cpu_mem_usage=True)
pipe = pipe.to("cpu")

prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt, num_inference_steps=31, width=400, height=400).images[0]

image.save("astrohorse-stable.png")
