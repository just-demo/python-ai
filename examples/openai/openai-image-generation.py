import base64
import uuid

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="Happy futuristic world",
    size="1024x1024",
    response_format="b64_json")
image_base64 = response.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

with open(f'image-{uuid.uuid4()}.png', 'wb') as f:
    f.write(image_bytes)
