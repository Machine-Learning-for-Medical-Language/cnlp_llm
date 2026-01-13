from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    Model,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .logprobs_generator import BatchedLogprobsGenerator


@solver
def seq_prob_multiple_choice(model: str | Model | None = None) -> Solver:
    # Defer initialization to avoid tokenizers parallelism warning
    generator: BatchedLogprobsGenerator | None = None

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nonlocal generator
        if generator is None:
            generator = BatchedLogprobsGenerator(get_model(model))

        if not state.choices:
            raise ValueError(
                "The seq_prob_multiple_choice solver requires samples with choices"
            )

        choice_messages: list[ChatMessage] = [
            ChatMessageAssistant(content=choice.value) for choice in state.choices
        ]

        choice_probs = await generator.get_choice_logprobs(
            prefix=state.messages, choices=choice_messages
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
