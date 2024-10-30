from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice


def get_medqa_dataset():
    def record_to_sample(record):
        return Sample(
            input=record["question"],
            choices=[record["options"][letter] for letter in "ABCD"],
            target=record["answer_idx"],
        )

    return hf_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="test",
        sample_fields=record_to_sample,
    )


@task
def medqa():
    return Task(
        dataset=get_medqa_dataset(),
        solver=multiple_choice(),
        scorer=choice(),
    )
