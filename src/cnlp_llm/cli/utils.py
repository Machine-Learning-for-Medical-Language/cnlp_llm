import traceback

import click

from ..eval import cnlp_dataset


def echo_error(e: Exception):
    exc_info = "".join(traceback.format_exception_only(e)).strip()
    click.echo(exc_info, err=True)
    return


def try_load_cnlp_dataset(dataset_file: str, task: str | None, **kwargs):
    try:
        return cnlp_dataset(dataset_file, task=task, **kwargs)
    except Exception as e:
        click.echo(
            f"Failed to load {dataset_file}. Currently only cnlp datasets are supported.",
            err=True,
        )
        echo_error(e)
        exit(1)
