from inspect_ai import Task, task
from inspect_ai.solver import generate, system_message


@task
def pirate_assistant(pirate_name: str = "Tensorbeard") -> Task:
    return Task(
        dataset=[],
        solver=[
            system_message(
                f"You are an AI assistant named {pirate_name} that always speaks like a pirate."
            ),
            generate(),
        ],
    )
