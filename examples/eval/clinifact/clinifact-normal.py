import sys

from cnlp_llm.eval import cnlp_dataset

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai import task
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message


### NOTE: This is NOT tournament style prompt.

# Neutral system message
SYSTEM_MESSAGE = """## Instruction:\nGiven a scientific claim and an abstract, determine if the abstract reports positive results (TRUE) or not (FALSE) about the claim.\nThe task is to classify the pair claim abstract as follows:\nTRUE: if the abstract provides positive support for the claim.\nFALSE: if the abstract provides negative or inconclusive support for the claim or if the abstract provides contextual or background information without directly reporting results about the claim.\n"""

# SYSTEM_MESSAGE = """
# Please read the following questions and decide whether it is "Yes" in a scientific manner (i.e., factual true). Don't explain your reasoning, just answer "Yes" or "No" on a new line.
# """


@task
def clinifact_task(dataset_file: str):
    return Task(
        dataset=cnlp_dataset(dataset_file, task="label"),
        plan=[system_message(SYSTEM_MESSAGE), generate(max_tokens=3)],
        scorer=match(location="begin"),
    )


if __name__ == "__main__":
    dataset_file = sys.argv[1]
    inspect_eval(
        clinifact_task,
        task_args={"dataset_file": dataset_file},
        # limit=100,  # uncomment this line to only evaluate 100 samples
    )
