import json
import pprint
import string

from openai import OpenAI
from dotenv import load_dotenv
from custom_letter_maths import int_to_letter, letter_to_int

load_dotenv(override=True)

client = OpenAI()


def build_training_prompt(a, b, c):
    return [
        {"role": "user", "content": f"{a}+{b}?"},
        {"role": "assistant", "content": c},
    ]


def generate_training_items():
    items = []
    for a in string.ascii_lowercase:
        for b in string.ascii_lowercase:
            c = int_to_letter(letter_to_int(a) + letter_to_int(b))
            items.append((a, b, c))
    return items


def generate_training_jsonl():
    messages = [json.dumps({"messages": build_training_prompt(a, b, c)}) for a, b, c in generate_training_items()]
    return "\n".join(messages)


def write_file(file, content):
    with open(file, "w") as f:
        f.write(content)


def upload_file(file):
    with open(file, "rb") as f:
        return client.files.create(file=f, purpose="fine-tune").id


write_file("openai_fine_tuning_letter_maths.jsonl", generate_training_jsonl())
training_file_id = upload_file("openai_fine_tuning_letter_maths.jsonl")
print(f"Uploaded file id: {training_file_id}")

job = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    # same structure as training_file, if you want to see validation metrics as well, like validation loss
    # validation_file=validation_file_id,
    model="gpt-4.1-nano-2025-04-14",
    hyperparameters={"n_epochs": 1},
    # TODO does this work? I don't see anything in https://wandb.ai/
    integrations=[{"type": "wandb", "wandb": {"project": "letter-maths"}}],
    suffix="maths")

print(f"Job id: {job.id}")
print(f"Fine-tuned model: {job.fine_tuned_model}")
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=10).data
pprint.pprint(events)
# Run openai_fine_tuning_progress.py to see the progress