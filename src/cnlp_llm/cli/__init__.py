import click
from inspect_ai._util.dotenv import init_dotenv

from .chat import chat
from .eval import eval
from .fine_tune import fine_tune
from .serve import serve


@click.group()
def cli():
    """Chat with, evaluate, and fine-tune LLMs."""
    init_dotenv()
    pass


cli.add_command(chat)
cli.add_command(eval)
cli.add_command(serve)
cli.add_command(fine_tune)


def main():
    cli()
