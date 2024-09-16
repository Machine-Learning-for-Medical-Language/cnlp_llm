from inspect_ai.solver import generate, system_message

from cnlp_llm.pipeline import Pipeline, servable
from cnlp_llm.pipeline.postprocess import extract_final_output


@servable
def get_pipeline(pirate_name: str = "Tensorbeard"):
    return Pipeline(
        name="pirate_assistant",
        plan=[
            system_message(
                f"You are an AI assistant named {pirate_name} that always speaks like a pirate."
            ),
            generate(),
        ],
        postprocess=extract_final_output,
    )


if __name__ == "__main__":
    pirate_assistant = get_pipeline()

    result = pirate_assistant(
        ["Hello!", "Who are you?", "What do you know about LLMs?"]
    )

    print(result)
