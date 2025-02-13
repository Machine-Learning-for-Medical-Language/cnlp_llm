from typing import Any

from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ...console import console
from .log import TournamentLog


class TournamentDisplay:
    def __init__(self, log: TournamentLog):
        self.log = log
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            auto_refresh=False,
            transient=True,
        )
        self.tournament_pbar = self.progress.add_task(
            "Tournament Progress:", total=log.n_rounds
        )
        self.round_pbar = None
        self.live = Live(self.panel, console=console)

    @property
    def title(self):
        return "LLM Tournament"

    @property
    def subtitle(self):
        return self.log.log_file

    @property
    def body(self):
        meta = Table.grid(padding=(0, 1))
        meta.add_column(style="blue", justify="right")
        meta.add_column()
        meta.add_row("Dataset:", self.log.dataset_path)
        meta.add_row("Samples:", str(self.log.samples))
        meta.add_row("Model:", self.log.model)
        meta.add_row("Scheduler:", self.log.scheduler)

        def _fmt(val: Any):
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        stats = Table.grid(padding=(0, 1))
        stats.add_column(style="blue", justify="right")
        stats.add_column()
        stats.add_row(
            "Metrics:",
            ", ".join(f"{k}={_fmt(v)}" for k, v in self.log.current_metrics.items()),
        )
        standings = list(self.log.current_standings.items())
        h_id, h_elo = standings[-1]
        l_id, l_elo = standings[0]
        stats.add_row("Highest ELO:", f"[b]{h_elo:.1f}[/b] (sample {h_id})")
        stats.add_row("Lowest ELO:", f"[b]{l_elo:.1f}[/b] (sample {l_id})")
        in_, out = self.log.model_usage()
        stats.add_row("Token Usage:", f"in={in_}, out={out}")

        body = Table.grid(expand=True)
        body.add_column()
        body.add_row(meta)
        body.add_row()
        body.add_row(self.progress)
        body.add_row()
        body.add_row(stats)

        return body

    @property
    def panel(self):
        panel = Panel(
            self.body,
            title="[bold]LLM Tournament[/bold]",
            subtitle=f"Log file: {self.log.log_file}",
            title_align="left",
            subtitle_align="left",
            padding=(1, 2),
            expand=True,
        )
        return panel

    def update(self):
        r, cur = self.log.current_round
        if cur:
            if self.round_pbar is None:
                self.round_pbar = self.progress.add_task(
                    f"Round {r}/{self.log.n_rounds}", total=cur.n_matches
                )

            self.progress.update(
                self.round_pbar,
                completed=cur.n_completed,
                total=cur.n_matches,
                description=f"Round {r}/{self.log.n_rounds}",
            )
            self.progress.update(
                self.tournament_pbar,
                completed=r - 1 + (cur.n_completed / cur.n_matches),
            )
        self.progress.refresh()
        self.live.update(self.panel)

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, *args):
        return self.live.__exit__(*args)
