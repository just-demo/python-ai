from time import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Just a demo")
audio_bytes = response.content

with open(f'audion-{round(time() * 1000)}.mp3', 'wb') as f:
    f.write(audio_bytes)
