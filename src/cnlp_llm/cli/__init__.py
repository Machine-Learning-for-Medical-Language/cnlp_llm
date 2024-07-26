import click
from inspect_ai._cli.main import inspect
from inspect_ai._util.dotenv import init_dotenv

from .chat import chat
from .fine_tune import fine_tune


@click.group()
def cli():
    init_dotenv()


cli.add_command(chat)

inspect.help = "Alias to the Inspect CLI for LLM evaluation."
inspect.name = "evaluation"
cli.add_command(inspect)

cli.add_command(fine_tune)
