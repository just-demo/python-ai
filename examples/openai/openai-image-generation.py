import base64

from time import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

# Just to prevent unintentional charges
if True:
    raise Exception("Each image generation request costs about 0.02 USD. Are you sure?")

response = client.images.generate(
    model="dall-e-2",
    prompt="Happy futuristic world",
    size="256x256",
    response_format="b64_json")
image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open(f'image-{round(time() * 1000)}.png', 'wb') as f:
    f.write(image_bytes)
