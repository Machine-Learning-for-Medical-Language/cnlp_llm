from inspect_ai.log import EvalSample
from inspect_ai.solver import generate, system_message

from cnlp_llm.infer import Pipeline


def extract_final_answer(sample: EvalSample):
    return sample.messages[-1].content


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
