from collections import Counter
from typing import Iterable, Literal

import tiktoken
from inspect_ai.dataset import Dataset
from inspect_ai.log import EvalLog
from inspect_ai.log._file import list_eval_logs, read_eval_log
from inspect_ai.model import ChatMessage
from rich.progress import track

from ...console import console


# Adapted from the openai cookbook
def count_tokens_from_messages(
    messages: list[ChatMessage], tiktoken_encoding_name="o200k_base"
) -> Counter[Literal["system", "user", "assistant", "tool"]]:
    """Return the approximate number of tokens used by a list of messages, categorized by role."""

    encoding = tiktoken.get_encoding(tiktoken_encoding_name)
    tokens_per_message = 3

    num_tokens = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
    for message in messages:
        role = message.role
        num_tokens[role] += tokens_per_message
        num_tokens[role] += len(encoding.encode(role))
        num_tokens[role] += len(encoding.encode(message.text))

    return Counter(num_tokens)  # type: ignore


def count_tokens_in_eval_log(
    log: EvalLog | str | None = None, tiktoken_encoding_name="o200k_base"
) -> Counter[Literal["system", "user", "assistant", "tool"]]:
    if log is None:
        log = list_eval_logs()[-1].name
    if isinstance(log, str):
        log = read_eval_log(log)
    counts = Counter({"system": 0, "user": 0, "assistant": 0, "tool": 0})
    for sample in log.samples or []:
        counts.update(
            count_tokens_from_messages(sample.messages, tiktoken_encoding_name)
        )
    return counts  # type: ignore


def get_dataset_token_counts(
    dataset: Dataset, tiktoken_encoding_name="o200k_base"
) -> Iterable[int]:
    try:
        encoding = tiktoken.get_encoding(tiktoken_encoding_name)
    except ValueError:
        raise ValueError(f"unknown encoding {tiktoken_encoding_name}")
    for sample in dataset:
        text_to_encode = str(
            sample.input
        )  # TODO support the case where this is a ChatMessage instead
        yield len(encoding.encode(text_to_encode))


def print_token_count_stats(dataset: Dataset, tiktoken_encoding_name="o200k_base"):
    n = total = min_ = max_ = 0
    for count in track(
        get_dataset_token_counts(dataset, tiktoken_encoding_name),
        "Tokenizing...",
        total=len(dataset),
    ):
        if n == 0:
            min_ = max_ = count

        n += 1
        total += count
        min_ = min(min_, count)
        max_ = max(max_, count)
    avg = total / n

    console.print(
        f"[b]{dataset.name}[/b] [i]{dataset.location}[i]",
        f"  Dataset size: [b cyan]{n:,}[/] samples",
        f"  Total tokens: [b cyan]{total:,}[/] ([i]{tiktoken_encoding_name}[/i])",
        f"  Min: [b cyan]{min_:,}",
        f"  Max: [b cyan]{max_:,}",
        f"  Avg: [b cyan]{avg:,.2f}",
        sep="\n",
        end="",
    )
    console.print()
