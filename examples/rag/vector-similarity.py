import os
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv(override=True)

hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def test_similarity(text1, text2):
    vector1, vector2 = model.encode([text1, text2])
    similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    print(f"{text1} + {text2} = {similarity * 100}%")


test_similarity("William Shakespeare", "Shakespeare")
test_similarity("William Shakespeare", "Shakespeare William")
test_similarity("William Shakespeare", "Shakespeare Will")
test_similarity("William Shakespeare", "Will Shakespeare")

test_similarity("William Shakespeare", "Hamlet")
test_similarity("William Shakespeare", "literature")
test_similarity("William Shakespeare", "boxing")

test_similarity("Muhammad Ali", "Hamlet")
test_similarity("Muhammad Ali", "literature")
test_similarity("Muhammad Ali", "boxing")

