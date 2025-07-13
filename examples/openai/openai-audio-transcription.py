from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

file = open("audio.mp3", "rb")
transcription = client.audio.transcriptions.create(model="whisper-1", file=file, response_format="text")
print(transcription)
