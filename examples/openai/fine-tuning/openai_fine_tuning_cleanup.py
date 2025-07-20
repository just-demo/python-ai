from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

jobs = {job.id: job.status for job in client.fine_tuning.jobs.list().data}
print(f"Cancelling jobs: {jobs}")
for job_id, job_status in jobs.items():
    if job_status not in ("cancelled", "failed", "succeeded"):
        client.fine_tuning.jobs.cancel(job_id)
        print(f"Cancelled job: {job_id} - {job_status}")

models = {model.id: model.owned_by for model in client.models.list().data if model.id.startswith("ft:")}
print(f"Deleting models: {models}")
for model_id, model_owned_by in models.items():
    client.models.delete(model_id)
    print(f"Deleted model: {model_id} - {model_owned_by}")

files = {file.id: file.filename for file in client.files.list().data}
print(f"Deleting files: {files}")
for file_id, file_name in files.items():
    client.files.delete(file_id)
    print(f"Deleted file: {file_id} - {file_name}")
