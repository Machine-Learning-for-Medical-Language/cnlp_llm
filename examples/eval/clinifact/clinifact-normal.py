import sys

from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

from cnlp_llm.eval import cnlp_dataset

### NOTE: This is NOT tournament style prompt.

# Neutral system message
SYSTEM_MESSAGE = """
## Instruction:\nGiven a scientific claim and an abstract, determine if the abstract reports positive results (TRUE) or not (FALSE) about the claim.\nThe task is to classify the pair claim abstract as follows:\nTRUE: if the abstract provides positive support for the claim.\nFALSE: if the abstract provides negative or inconclusive support for the claim or if the abstract provides contextual or background information without directly reporting results about the claim.
"""
SYSTEM_MESSAGE = " ".join(SYSTEM_MESSAGE.splitlines())


@task
def clinifact_task(dataset_file: str):
    return Task(
        dataset=cnlp_dataset(dataset_file, task="label"),
        plan=[system_message(SYSTEM_MESSAGE), generate(max_tokens=6)],
        # scorer=includes(),
        scorer=match(location="any"),
    )


if __name__ == "__main__":
    dataset_file = sys.argv[1]
    inspect_eval(
        clinifact_task,
        task_args={"dataset_file": dataset_file},
        log_format="json",
        # limit=100,  # uncomment this line to only evaluate 100 samples
    )
