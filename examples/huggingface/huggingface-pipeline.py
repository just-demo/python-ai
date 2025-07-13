import os
import logging
import torch
from time import time
from dotenv import load_dotenv
from huggingface_hub import login
from transformers.utils import logging as hf_logging
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf

load_dotenv(override=True)

# Uncomment this to enable debugging
# logging.basicConfig(level=logging.DEBUG)
# hf_logging.set_verbosity_debug()

hf_token = os.getenv("HF_TOKEN")
# This persists the credentials in osxkeychain to be futher used to check out code from huggingface.co
login(hf_token, add_to_git_credential=True)

# Sentiment Analysis
sentiment_analyzer = pipeline("sentiment-analysis")
result = sentiment_analyzer("I ma so happy!")
print(result)

# Named Entity Recognition
entity_recognizer = pipeline("ner", grouped_entities=True)
result = entity_recognizer("Bob lived in London")
print(result)

# Question Answering
answerer = pipeline("question-answering")
result = answerer(question="Who lives in London?", context="Bob lives in London")
print(result)

# Translation
translator = pipeline("translation_en_to_fr")
result = translator("Bob lives in London")
print(result[0]["translation_text"])

# Classification
classifier = pipeline("zero-shot-classification")
result = classifier("William Shakespeare was an English playwright",
                    candidate_labels=["gaming", "literature", "history"])
print(result)

# Text Generation
text_generator = pipeline("text-generation")
result = text_generator("The most important thing is")
print(result[0]["generated_text"])

# Summarization
text = """William Shakespeare was an English playwright, poet and actor. He is widely regarded as the greatest writer
in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the "Bard
of Avon" or simply "the Bard". His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three
long narrative poems and a few other verses, some of uncertain authorship. His plays have been translated into every
major living language and are performed more often than those of any other playwright. Shakespeare remains arguably the
most influential writer in the English language, and his works continue to be studied and reinterpreted.
"""
# TODO: this failed with: we now require users to upgrade torch to at least v2.6
summarizer = pipeline("summarization")
result = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(result[0]["summary_text"])

# Image Generation
# TODO: this takes long time - does this work?
image_gen = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16").to("cpu")
image = image_gen(prompt="Happy futuristic world").images[0]
image.save(f"image_{round(time() * 1000)}.png")

# Audio Generation
# TODO: this takes long time - does this work?
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
speech = synthesiser("Who are you?", forward_params={"speaker_embeddings": speaker_embedding})
sf.write(f"audio_{round(time() * 1000)}.wav", speech["audio"], samplerate=speech["sampling_rate"])
