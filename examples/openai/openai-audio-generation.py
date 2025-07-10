import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Just a demo")
audio_bytes = response.content

with open(f'audion-{uuid.uuid4()}.mp3', 'wb') as f:
    f.write(audio_bytes)
