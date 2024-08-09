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
