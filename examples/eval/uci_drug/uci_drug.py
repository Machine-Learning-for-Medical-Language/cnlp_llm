import sys

from cnlp_llm.eval import cnlp_dataset

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai import task
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

SYSTEM_MESSAGE = """
Please read the following drug review and rate the satisfaction as Low, Medium, or High.
Don't explain your reasoning, just respond with "Low", "Medium", or "High" on a new line.
"""


@task
def uci_drug_task(dataset_file: str):
    return Task(
        dataset=cnlp_dataset(dataset_file, task="sentiment"),
        plan=[system_message(SYSTEM_MESSAGE), generate(max_tokens=3)],
        scorer=match(location="begin"),
    )


if __name__ == "__main__":
    dataset_file = sys.argv[1]
    inspect_eval(
        uci_drug_task,
        task_args={"dataset_file": dataset_file},
        # limit=100,  # uncomment this line to only evaluate 100 samples
    )
