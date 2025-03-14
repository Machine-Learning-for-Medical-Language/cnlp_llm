from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    Model,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .logprobs_generator import BatchedLogprobsGenerator

TEMPLATE = """
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letter_choices}.

{question}

{answer_choices}
""".strip()


def make_prompt(question: str, choices: list[tuple[str, str]]):
    letters = [c[0] for c in choices]
    letter_choices = ", ".join(letters[:-1]) + ", or " + letters[-1]
    answer_choices = "\n".join(f"{c[0]}) {c[1]}" for c in choices)
    return TEMPLATE.format(
        letter_choices=letter_choices, question=question, answer_choices=answer_choices
    )


@solver
def letter_prob_multiple_choice(model: str | Model | None = None) -> Solver:
    # Defer initialization to avoid tokenizers parallelism warning
    generator: BatchedLogprobsGenerator | None = None

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nonlocal generator
        if generator is None:
            generator = BatchedLogprobsGenerator(get_model(model))

        if not state.choices:
            raise ValueError(
                "The letter_prob_multiple_choice solver requires samples with choices"
            )

        letters = [chr(ord("A") + i) for i in range(len(state.choices))]
        choice_values = [choice.value for choice in state.choices]

        state.user_prompt.text = make_prompt(
            question=state.user_prompt.text, choices=list(zip(letters, choice_values))
        )

        choice_messages: list[ChatMessage] = [
            ChatMessageAssistant(content=f"ANSWER: {letter}") for letter in letters
        ]

        choice_probs = await generator.compare_logprobs(
            choice_messages, prepend=state.messages
        )

        ranked = sorted(
            zip(state.choices, choice_probs, choice_messages),
            key=lambda tup: tup[1],
            reverse=True,
        )
        chosen, *rejected = [choice for choice, prob, letter in ranked]

        chosen.correct = True
        for c in rejected:
            c.correct = False

        state.messages.append(ranked[0][2])

        # update metadata with probabilities values
        if "letter_probs" not in state.metadata:
            state.metadata["letter_probs"] = []
        seq_prob_metadata: list[dict[str, float]] = state.metadata["letter_probs"]
        probs_map = {letter: prob for letter, prob in zip(letters, choice_probs)}
        seq_prob_metadata.append(probs_map)

        return state

    return solve
