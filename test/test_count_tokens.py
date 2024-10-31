from inspect_ai.dataset import MemoryDataset, Sample

from cnlp_llm.eval import get_dataset_token_counts


def test_count_tokens():
    n = 5

    single_token = " token"
    samples = [Sample(input=single_token * i) for i in range(1, n + 1)]
    dataset = MemoryDataset(samples)

    counts = list(
        get_dataset_token_counts(dataset, tiktoken_encoding_name="o200k_base")
    )

    assert len(counts) == n

    for i, count in enumerate(counts, 1):
        assert count == i
