from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .common import get_choices_logprobs


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
                reverse=True,
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
