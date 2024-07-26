import subprocess

import click
from trl.commands.scripts import sft as trl_sft
from trl.commands.scripts import dpo as trl_dpo


@click.group("finetune")
def fine_tune():
    """Fine-tune a model."""
    pass


@fine_tune.command(
    "sft",
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
@click.argument("sft_args", nargs=-1, type=click.UNPROCESSED)
def sft(sft_args):
    "Supervised fine-tuning via `trl sft`"
    command = ["python", trl_sft.__file__] + list(sft_args)
    subprocess.run(command)


@fine_tune.command(
    "dpo",
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
@click.argument("dpo_args", nargs=-1, type=click.UNPROCESSED)
def dpo(dpo_args):
    "DPO fine-tuning via `trl dpo`"
    command = ["python", trl_dpo.__file__] + list(dpo_args)
    subprocess.run(command)
