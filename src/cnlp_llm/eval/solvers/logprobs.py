import torch
from inspect_ai.model import ChatMessage, ChatMessageAssistant, Model, get_model
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.model._providers.util.chatapi import chat_api_input
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolInfo


@solver
def seq_prob_multiple_choice() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("The multiple_choice solver requires samples with choices")

        model = get_model(str(state.model))

        choice_probs: list[float] = []
        for choice in state.choices:
            choice_probs.append(
                await get_sequence_logprob(
                    model, state.messages + [ChatMessageAssistant(content=choice.value)]
                )
            )

        chosen, *rejected = [
            choice
            for choice, prob in sorted(
                zip(state.choices, choice_probs),
                key=lambda tup: tup[1],
            )
        ]

        chosen.correct = True
        for c in rejected:
            c.correct = False

        state.messages.append(ChatMessageAssistant(content=chosen.value))

        return state

    return solve


async def get_sequence_logprob(
    inspect_model: Model,
    messages: list[ChatMessage],
    tools: list[ToolInfo] | None = None,
) -> float:
    model_api = inspect_model.api
    if not isinstance(model_api, HuggingFaceAPI):
        raise ValueError(
            "Only the Huggingface API is supported for evaluating sequence probabilities."
        )

    model = model_api.model
    tokenizer = model_api.tokenizer
    chat_template = model_api.chat_template

    # convert to hf format
    hf_messages = chat_api_input(messages, tools or [])

    # apply chat template
    input_tokens: torch.Tensor = tokenizer.apply_chat_template(
        hf_messages,  # type: ignore
        chat_template=chat_template,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits: torch.Tensor = model(input_tokens.to(model.device)).logits
        logprobs = torch.log_softmax(logits, -1).squeeze()[:-1]
        logprobs = logprobs[torch.arange(logprobs.shape[0]), input_tokens.squeeze()[1:]]
        sequence_probability = logprobs.sum().item()

    return sequence_probability
