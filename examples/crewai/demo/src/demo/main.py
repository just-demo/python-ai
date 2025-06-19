#!/usr/bin/env python
import warnings

from demo.crew import Demo

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    try:
        result = Demo().crew().kickoff(inputs={
            'idea': 'Buy an apartment in Paris'
        })
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    print("Training...")


def replay():
    print("Replaying...")


def test():
    print("Testing...")
