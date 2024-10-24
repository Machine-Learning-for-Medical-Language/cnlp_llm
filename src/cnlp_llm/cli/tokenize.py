import traceback

import click

from ..eval import cnlp_dataset, print_token_count_stats


@click.command()
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
def tokenize(
    dataset_file: str,
):
    "Get token counts for a dataset."
    try:
        dataset = cnlp_dataset(dataset_file, task=None)
    except Exception as e:
        click.echo(
            f"Failed to load {dataset_file}. Currently only cnlp datasets are supported.",
            err=True,
        )
        exc_info = "".join(traceback.format_exception_only(e)).strip()
        click.echo(exc_info, err=True)
        return

    print_token_count_stats(dataset)
