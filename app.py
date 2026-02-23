from diffusers import StableDiffusionPipeline
from PIL import Image

pipe = StableDiffusionPipeline.from_pretrained("~/Models/stable-diffusion-v1-5", low_cpu_mem_usage=True)
pipe = pipe.to("cpu")

prompt = "an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=5, width=640, height=640).images[0]

image.save("astronaut_horse.png")