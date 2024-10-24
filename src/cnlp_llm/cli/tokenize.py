import traceback

import click

from ..eval import cnlp_dataset, print_token_count_stats


def echo_error(e: Exception):
    exc_info = "".join(traceback.format_exception_only(e)).strip()
    click.echo(exc_info, err=True)
    return


@click.command()
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--tiktoken-encoding-name",
    help="name of the tiktoken encoder to use for tokenizing.",
    type=str,
    default="o200k_base",
)
def tokenize(
    dataset_file: str,
    tiktoken_encoding_name: str,
):
    """
    Get token counts for DATASET_FILE.

    DATASET_FILE must be a path to a single file that is a valid cnlp dataset
    loadable with cnlp_llm.eval.cnlp_dataset().

    The --tiktoken-encoding-name option chooses the encoding for tiktoken to use for tokenization.
    This defaults to "o200k_base", which is the tokenizer used by gpt-4o and gpt-4o-mini. For more
    options, see the OpenAI cookbook here:

    \b
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    """
    try:
        dataset = cnlp_dataset(dataset_file, task=None)
    except Exception as e:
        click.echo(
            f"Failed to load {dataset_file}. Currently only cnlp datasets are supported.",
            err=True,
        )
        echo_error(e)
        exit(1)
    try:
        print_token_count_stats(dataset, tiktoken_encoding_name)
    except ValueError as e:
        echo_error(e)
        exit(1)
