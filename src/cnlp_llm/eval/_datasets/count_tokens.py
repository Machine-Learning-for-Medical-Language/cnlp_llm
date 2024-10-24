from typing import Iterable

import tiktoken
from inspect_ai.dataset import Dataset
from rich.progress import track

from ...console import console


def get_token_counts(
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
        get_token_counts(dataset, tiktoken_encoding_name),
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
