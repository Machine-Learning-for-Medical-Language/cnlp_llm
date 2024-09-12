from inspect_ai.solver import generate, system_message

from cnlp_llm.pipeline import Pipeline, servable
from cnlp_llm.pipeline.postprocess import extract_final_output


@servable(log_dir="logs/server")
def get_pipeline():
    return Pipeline(
        name="pirate_assistant",
        plan=[
            system_message("You are an AI assistant that always speaks like a pirate."),
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
