from inspect_ai.solver import generate, system_message

from cnlp_llm.pipeline import Pipeline
from cnlp_llm.pipeline.postprocess import extract_final_answer


if __name__ == "__main__":

    pirate_assistant = Pipeline(
        name="pirate_assistant",
        plan=[
            system_message("You are an AI assistant that always speaks like a pirate."),
            generate(),
        ],
        postprocess=extract_final_answer,
    )

    result = pirate_assistant(
        ["Hello!", "Who are you?", "What do you know about LLMs?"]
    )

    print(result)
