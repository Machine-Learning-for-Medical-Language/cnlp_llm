import sys

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

from cnlp_llm.eval import cnlp_dataset

### NOTE: This is NOT tournament style prompt.

# Neutral system message
SYSTEM_MESSAGE = """
You are a biomedical expert. Given a yes/no question related to biomedical science, determine whether the answer is "Yes" based strictly on scientific facts. Do not provide any explanations, just output "Yes" or "No" on a new line.
"""

# SYSTEM_MESSAGE = """
# Please read the following questions and decide whether it is "Yes" in a scientific manner (i.e., factual true). Don't explain your reasoning, just answer "Yes" or "No" on a new line.
# """


@task
def bioasq12b_task(dataset_file: str):
    return Task(
        dataset=cnlp_dataset(dataset_file, task="label"),
        plan=[system_message(SYSTEM_MESSAGE), generate(max_tokens=3)],
        scorer=match(location="begin"),
    )


if __name__ == "__main__":
    dataset_file = sys.argv[1]
    inspect_eval(
        bioasq12b_task,
        task_args={"dataset_file": dataset_file},
        # limit=100,  # uncomment this line to only evaluate 100 samples
    )
