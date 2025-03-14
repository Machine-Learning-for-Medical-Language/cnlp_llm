import sys

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

from cnlp_llm.eval import cnlp_dataset

# Neutral system message
SYSTEM_MESSAGE = """
Please read the following sentence and decide whether it is "acceptable" in a linguistic sense (i.e., grammatical). Don't explain your reasoning, just answer "Yes" (acceptable) or "No" (unacceptable) on a new line.
"""

# System message favoring precision
# SYSTEM_MESSAGE = """
# Please read the following sentence and decide whether it is "acceptable" in a linguistic sense (i.e., grammatical). Don't explain your reasoning, just answer "Yes" (acceptable) or "No" (unacceptable) on a new line. The consequences for wrongly guessing "Yes" are worse than the consequences for wrongly guessing "No".
# """

# System message favoring recall
# SYSTEM_MESSAGE = """
# Please read the following sentence and decide whether it is "acceptable" in a linguistic sense (i.e., grammatical). Don't explain your reasoning, just answer "Yes" (acceptable) or "No" (unacceptable) on a new line. The consequences for wrongly guessing "No" are worse than the consequences for wrongly guessing "Yes".
# """


@task
def cola_task(dataset_file: str):
    return Task(
        dataset=cnlp_dataset(dataset_file, task="acceptable"),
        plan=[system_message(SYSTEM_MESSAGE), generate(max_tokens=3)],
        scorer=match(location="begin"),
    )


if __name__ == "__main__":
    dataset_file = sys.argv[1]
    inspect_eval(
        cola_task,
        task_args={"dataset_file": dataset_file},
        # limit=100,  # uncomment this line to only evaluate 100 samples
    )
