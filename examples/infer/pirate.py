from inspect_ai.solver import TaskState, generate, system_message

from cnlp_llm.infer import execute_plan


if __name__ == "__main__":

    async def get_final_output(state: TaskState):
        return state.output.message.content

    prompts = ["Hello!", "Who are you?", "What do you know about LLMs?"]
    plan = [
        system_message("You are an AI assistant that always speaks like a pirate."),
        generate(),
    ]
    result = execute_plan(prompts, plan, postprocess=get_final_output)
    print(result)
