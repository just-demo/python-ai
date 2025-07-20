from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

models = client.models.list().data
nano_model_ids = [m.id for m in models if 'nano' in m.id]

print(nano_model_ids)
