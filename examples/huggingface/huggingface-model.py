import os

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from transformers.utils import logging as hf_logging
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.DEBUG)
hf_logging.set_verbosity_debug()

hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "google/gemma-2-2b-it"

messages = [
    {"role": "system", "content": "Respond to user"},
    {"role": "user", "content": "Who are you?"}
]

# TODO: This doesn't work on Mac (no bitsandbytes>=0.43.1 supported by Mac):
#  The installed version of bitsandbytes (<0.43.1) requires CUDA, but CUDA is not available.
#  You may need to install PyTorch with CUDA support or upgrade bitsandbytes to >=0.43.1.
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cpu")
print(inputs)

# TODO: It takes long time, does it work?
model = AutoModelForCausalLM.from_pretrained(model_name) #, quantization_config=quantization_config)
print(model)
print(f"Model memory: {model.get_memory_footprint() / 1e6} MB")

outputs = model.generate(inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0]))

