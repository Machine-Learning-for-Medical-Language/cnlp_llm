import click
from inspect_ai._cli.main import inspect
from inspect_ai._util.dotenv import init_dotenv

from .chat import chat


@click.group()
def cli():
    init_dotenv()


cli.add_command(chat)

inspect.help = "Alias for the Inspect CLI."
cli.add_command(inspect)
