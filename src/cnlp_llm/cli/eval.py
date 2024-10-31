import subprocess

import click
from inspect_ai.model._providers.mockllm import MockLLM

from cnlp_llm.console import console
from cnlp_llm.eval._datasets.count_tokens import count_tokens_in_eval_log


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("eval_args", nargs=-1, type=click.UNPROCESSED)
def eval(eval_args):
    """
    LLM evaluation via `inspect eval`.

    Use --dry-run to run the task with a dummy model and count tokens.
    """
    args = list(eval_args)
    if len(args) == 0:
        args = ["--help"]

    dry_run = False
    if "--dry-run" in args:
        dry_run = True
        args.remove("--dry-run")
        args.extend(["--model", "mockllm/model"])
        if "-y" in args:
            args.remove("-y")
        else:
            console.print(
                "This dry run will run the task with `--model mockllm/model` and count tokens."
            )
            console.print(
                f'The MockLLM model is a dummy model that always outputs "{MockLLM.default_output}".'
            )
            console.print(
                "[yellow]If your task explicitly uses a different model, that model will still be used."
            )
            console.print("Run with `-y` to silence this warning.")
            choice = console.input("Begin dry run? (Y/n): ")
            if not (choice.lower() == "y" or choice == ""):
                return

    command = ["inspect", "eval"] + args
    subprocess.run(command)

    if dry_run:
        console.print("Dry run complete.")
        counts = count_tokens_in_eval_log()
        console.print("Token counts:")
        for role in ["system", "user", "assistant", "tool"]:
            console.print(f"  {role.ljust(9)} - {counts[role]}")
        console.print(f"  [b]{'total'.ljust(9)}[/] - {counts.total()}")
