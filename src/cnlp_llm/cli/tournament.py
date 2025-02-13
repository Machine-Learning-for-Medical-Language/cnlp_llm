import click

from ..eval.tournament import ComparisonPrompt, Tournament
from ..eval.tournament.scheduler import (
    GraphScheduler,
    OutsideInScheduler,
    RandomScheduler,
    SlidingScheduler,
    SwissScheduler,
)
from .utils import try_load_cnlp_dataset

SCHEDULERS = {
    "random": RandomScheduler,
    "outside-in": OutsideInScheduler,
    "sliding": SlidingScheduler,
    "swiss": SwissScheduler,
    "graph": GraphScheduler,
}


@click.command()
@click.argument("dataset_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("comparison_prompt_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--task", type=str, required=True, help="The task (dataset column) to target."
)
@click.option(
    "--pos-label",
    type=str,
    required=True,
    help="The label corresponding to positive instances in the dataset.",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    type=str,
    required=True,
    envvar=["INSPECT_EVAL_MODEL"],
    help="Model used to run the tournament. Can also be set via INSPECT_EVAL_MODEL environment variable.",
)
@click.option(
    "--rounds", type=int, required=True, help="The number of tournament rounds to run."
)
@click.option(
    "--scheduler",
    type=click.Choice(list(SCHEDULERS.keys()), case_sensitive=False),
    default="graph",
    help="Tournament scheduler to use.",
)
@click.option(
    "--log-dir",
    type=str,
    required=True,
    envvar=["CNLP_TOURNAMENT_LOG_DIR"],
    help="Directory to write the tournament log.",
)
@click.option(
    "--limit", type=int, help="Only use a subset of the dataset for the tournament."
)
def tournament(
    dataset_file: str,
    comparison_prompt_file: str,
    task: str,
    pos_label: str,
    model_name: str,
    rounds: int,
    scheduler: str,
    log_dir: str,
    limit: int | None = None,
):
    dataset = try_load_cnlp_dataset(dataset_file, task)
    comparison_prompt = ComparisonPrompt.from_file(comparison_prompt_file)
    tournament = Tournament(
        model=model_name,
        dataset=dataset,
        sample_to_binary=lambda s: s.target == pos_label,
        comparison_prompt=comparison_prompt,
        log_dir=log_dir,
        max_players=limit,
    )

    tournament.run(rounds=rounds, scheduler=SCHEDULERS[scheduler]())
