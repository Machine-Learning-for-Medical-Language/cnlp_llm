# Evaluation Examples

> [!TIP]
Many inference parameters can be automatically loaded from a `.env` file in your working directory. See the [example `.env` file](../../.env.example) for an example with links to the Inspect documentation on relevant environment variables.
>
> To create your own `.env` file based on the example, run `cp .env.example .env` from the base directory, and modify the result as needed.
>
> To choose your model for evaluation, set the `INSPECT_EVAL_MODEL` variable to a valid [model identifier](../chat/README.md#example-model-identifiers).

## Security Guide

Copied from [here](https://inspect.ai-safety-institute.org.uk/examples.html#sec-security-guide).

Run with `python examples/security_guide.py` or `inspect eval examples/security_guide.py`

## UCI Drug Review Sentiment Classification

1. Download the [Drug Reviews (Drugs.com) data set](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) and extract. Pay attention to their terms:

    > - Only use the data for research purposes
    > - Don't use the data for any commerical purposes
    > - Don't distribute the data to anyone else
    > - Cite us

2. Preprocess the data

    ```bash
    python examples/eval/uci_drug/prepare_data.py <path/to/extracted/data> <path/to/output>
    ```

3. Make sure you have configured a model for evaluation (see tip above).
4. Run the evaluation

    ```bash
    python examples/eval/uci_drug/uci_drug.py <path/to/preprocessed/data/dev.csv>
    ```

## CoLA Classification Tournament

Using an LLM as a binary classifier can be effective, but unlike traditional classifiers,
they make binary classifications discretely, rather than assigning a 0–1 probability score.
This can make it hard to use traditional classifier evaluation metrics such as AUROC.

This example demonstrates a way to convert discrete binary predictions to probability scores
by running a tournament where samples from a dataset are put head-to-head to determine
which one is 'more positive'. In practice, this is essentially what we ask the LLM: "which of
these two samples is more positive based on our criteria?"

Each sample has an [ELO score](https://en.wikipedia.org/wiki/Elo_rating_system), which is
updated over the course of the tournament. When the tournament is over, we can convert
these ELO scores to probabilities.

The example in `examples/eval/cola` runs a tournament with the
[CoLA dataset](https://nyu-mll.github.io/CoLA/), classifying sentences as linguistically
acceptable or unacceptable. You can preprocess the data with `preprocess_cola.py`,
and run the tournament with `cola_tournament.py path/to/preprocessed/cola/in_domain_dev.tsv`.

Alternatively you can use the command line:

```sh
cnlp_llm tournament path/to/preprocessed/cola/in_domain_dev.tsv \
    examples/eval/cola/cola.prompt \
    --task acceptable \
    --pos-label Yes \
    --model ollama/llama3.2:1b \
    --rounds 10 \
    --scheduler graph
```
