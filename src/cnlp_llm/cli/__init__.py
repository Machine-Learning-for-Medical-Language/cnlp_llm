import click
from inspect_ai._util.dotenv import init_dotenv

from .chat import chat
from .fine_tune import fine_tune
from .serve import serve


@click.group()
def cli():
    """Chat with, evaluate, and fine-tune LLMs."""
    init_dotenv()
    pass


@cli.command()
def eval():
    """LLM evaluation."""
    click.echo(
        "For evaluation, use the `inspect` command.\nFor more information, see https://inspect.ai-safety-institute.org.uk/workflow.html"
    )


cli.add_command(chat)
cli.add_command(serve)
cli.add_command(fine_tune)


def main():
    cli()
