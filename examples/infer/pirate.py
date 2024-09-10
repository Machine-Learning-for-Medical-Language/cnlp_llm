from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import TaskState, generate, system_message

from cnlp_llm.infer import execute_plan


if __name__ == "__main__":

    async def get_final_output(state: TaskState):
        return state.output.message.content

    prompts = ["Hello!", "Who are you?", "What do you know about LLMs?"]

    dataset = MemoryDataset(samples=[Sample(input=prompt) for prompt in prompts])
    plan = [
        system_message("You are an AI assistant that always speaks like a pirate."),
        generate(),
    ]
    result = execute_plan(dataset, plan, postprocess=get_final_output)
    print(result)
