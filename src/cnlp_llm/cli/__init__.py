import click

from .annotation_calculator import annotation_calculator
from .chat import chat
from .completions import completions
from .eval import eval
from .fine_tune import fine_tune
from .serve import serve
from .tokenize import tokenize
from .tournament import tournament


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    """Chat with, evaluate, and fine-tune LLMs."""
    if ctx.invoked_subcommand is not None:
        from inspect_ai._util.dotenv import init_dotenv

        init_dotenv()


cli.add_command(annotation_calculator)
cli.add_command(chat)
cli.add_command(completions)
cli.add_command(eval)
cli.add_command(serve)
cli.add_command(fine_tune)
cli.add_command(tokenize)
cli.add_command(tournament)


def main():
    cli()
