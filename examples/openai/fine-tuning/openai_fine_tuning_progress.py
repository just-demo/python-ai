import pprint

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

jobs = client.fine_tuning.jobs.list().data
job = sorted(jobs, key=lambda job: job.created_at)[-1]
print(f"Job id: {job.id}")
print(f"Fine-tuned model: {job.fine_tuned_model}")
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=10).data
pprint.pprint(events)
