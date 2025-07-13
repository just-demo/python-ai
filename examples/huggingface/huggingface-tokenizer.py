import os

import logging
from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.DEBUG)
hf_logging.set_verbosity_debug()

hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
text = "William Shakespeare was an English playwright, poet and actor."
tokens = tokenizer.encode(text)
print(len(text))
print(len(tokens))
print(tokens)
print(tokenizer.decode(tokens))
print(tokenizer.batch_decode(tokens))
print(tokenizer.get_added_vocab())

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
messages = [
    {"role": "system", "content": "Respond to user"},
    {"role": "user", "content": "Who are you?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
