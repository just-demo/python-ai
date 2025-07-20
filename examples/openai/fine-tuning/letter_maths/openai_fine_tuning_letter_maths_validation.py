import string

from openai import OpenAI
from dotenv import load_dotenv
from custom_letter_maths import letter_to_int, int_to_letter

load_dotenv(override=True)

base_model = "gpt-4.1-nano"
fine_tuned_model = "ft:gpt-4.1-nano-2025-04-14:personal:maths:<replace-with-real-name>"
client = OpenAI()


def ask(model, question):
    messages = [{"role": "user", "content": question}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def validate(model, question):
    answer = ask(model, question)
    print(f"{question.rstrip("?")}={answer}")


def validate_all(model):
    print("-" * 20)
    count_all = 0
    count_failed = 0
    for a in string.ascii_lowercase:
        for b in string.ascii_lowercase:
            count_all = count_all + 1
            c = int_to_letter(letter_to_int(a) + letter_to_int(b))
            answer = ask(model, f"{a}+{b}?")
            if answer != c:
                count_failed += 1
                print(f"{a}+{b}={c} <> {answer}")
    print(f"Failed: {count_failed}/{count_all}")


validate(base_model, "a+b?")
validate(fine_tuned_model, "a+b?")
validate(fine_tuned_model, "ab+cd?")
validate(fine_tuned_model, "abc+def?")
validate(fine_tuned_model, "hello+world?")
# validate_all(fine_tuned_model) # Failed: 218/676
