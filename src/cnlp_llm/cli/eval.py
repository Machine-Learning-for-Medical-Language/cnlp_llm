import subprocess

import click


@click.command()
@click.argument("eval_args", nargs=-1, type=click.UNPROCESSED)
def eval(eval_args):
    """LLM evaluation."""
    args = list(eval_args)
    if len(args) == 0:
        args = ["--help"]

    command = ["inspect", "eval"] + args
    subprocess.run(command)
