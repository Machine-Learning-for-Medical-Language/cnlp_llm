import torch
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    Model,
    get_model,
)
from inspect_ai.model._providers.hf import HuggingFaceAPI
from inspect_ai.model._providers.util.chatapi import chat_api_input
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolInfo
from transformers import PreTrainedModel


@solver
def seq_prob_multiple_choice() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("The multiple_choice solver requires samples with choices")

        model = get_model(str(state.model))

        choice_messages: list[ChatMessage] = [
            ChatMessageAssistant(content=choice.value) for choice in state.choices
        ]

        choice_probs: list[float] = await get_choices_logprobs(
            model, prefix_messages=state.messages, choices=choice_messages
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

        # update metadata with probabilities values
        if "seq_probs" not in state.metadata:
            state.metadata["seq_probs"] = []
        seq_prob_metadata: list[dict[str, float]] = state.metadata["seq_probs"]
        probs_map = {
            choice.value: prob for choice, prob in zip(state.choices, choice_probs)
        }
        seq_prob_metadata.append(probs_map)

        return state

    return solve


async def get_choices_logprobs(
    inspect_model: Model,
    prefix_messages: list[ChatMessage],
    choices: list[ChatMessage],
    tools: list[ToolInfo] | None = None,
) -> list[float]:
    model_api = inspect_model.api
    if not isinstance(model_api, HuggingFaceAPI):
        raise ValueError(
            "Only the Huggingface API is supported for evaluating sequence probabilities."
        )

    model: PreTrainedModel = model_api.model
    tokenizer = model_api.tokenizer
    chat_template = model_api.chat_template

    # convert inspect ChatMessages to huggingface format
    hf_choices_messages = [
        chat_api_input(prefix_messages + [choice], tools or []) for choice in choices
    ]

    batched_tokens_input: torch.Tensor = tokenizer.apply_chat_template(
        hf_choices_messages,  # type: ignore
        chat_template=chat_template,
        return_tensors="pt",
        padding=True,
    ).to(model.device)  # type: ignore

    sequence_probabilities: list[float] = []

    with torch.inference_mode():
        logits = model(
            batched_tokens_input,
        ).logits
        logprobs = torch.log_softmax(logits, -1)[:, :-1, :]
        logprobs = logprobs.gather(-1, batched_tokens_input[:, 1:, None]).squeeze()
        sequence_probabilities = logprobs.sum(dim=-1).tolist()

    return sequence_probabilities
