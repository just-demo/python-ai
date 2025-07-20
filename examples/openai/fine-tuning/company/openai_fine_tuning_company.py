import json
import pprint

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()


def read_questions(file, name_replacement) -> dict[str, str]:
    questions = {}
    with open(file, "r") as f:
        lines = f.read().split("\n")
        for index, line in enumerate(lines):
            if line.startswith("Q: "):
                questions[line.removeprefix("Q: ")] = lines[index + 1].removeprefix("A: ").replace("{name}",
                                                                                                   name_replacement)
    return questions


def build_training_jsonl(questions):
    lines = [{"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]} for question, answer in questions.items()]
    return "\n".join([json.dumps(line) for line in lines])


def write_file(file, content):
    with open(file, "w") as f:
        f.write(content)


def upload_file(file):
    with open(file, "rb") as f:
        return client.files.create(file=f, purpose="fine-tune").id


questions = read_questions("openai_fine_tuning_company.txt", "JustDemo")
write_file("openai_fine_tuning_company.jsonl", build_training_jsonl(questions))
training_file_id = upload_file("openai_fine_tuning_company.jsonl")
print(f"Uploaded file id: {training_file_id}")

job = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    # same structure as training_file, if you want to see validation metrics as well, like validation loss
    # validation_file=validation_file_id,
    model="gpt-4.1-nano-2025-04-14",
    hyperparameters={"n_epochs": 1},
    # TODO does this work? I don't see anything in https://wandb.ai/
    integrations=[{"type": "wandb", "wandb": {"project": "company-info"}}],
    suffix="company")

print(f"Job id: {job.id}")
print(f"Fine-tuned model: {job.fine_tuned_model}")
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=10).data
pprint.pprint(events)
# Run openai_fine_tuning_progress.py to see the progress
