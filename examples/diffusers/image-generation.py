import time

from diffusers import StableDiffusionPipeline
import logging
from transformers.utils import logging as hf_logging
import torch

logging.basicConfig(level=logging.DEBUG)
hf_logging.set_verbosity_debug()

timestamp = {"started": time.time()}

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, ).to("cpu")

timestamp['downloaded'] = time.time()

prompt = "Happy futuristic world"
image = pipe(prompt).images[0]
image.save(f"generated_image_{str(round(time.time() * 1000))}.png")

timestamp['generated'] = time.time()

print(f"Download time: {timestamp['downloaded'] - timestamp['started']} seconds")
print(f"Generate time: {timestamp['generated'] - timestamp['downloaded']} seconds")
