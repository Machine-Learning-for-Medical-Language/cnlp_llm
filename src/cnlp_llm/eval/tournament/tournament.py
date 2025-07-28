import random
from collections.abc import Callable
from typing import Any

import anyio
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import Model, get_model
from inspect_ai.util import collect
from shortuuid import uuid

from .display import TournamentDisplay
from .log import MatchResult, TournamentLog
from .metrics import get_best_accuracy, get_best_f1, roc_auc_score
from .player import Player
from .prompt import ComparisonPrompt
from .scheduler import Scheduler


def elo_init_constant(sample: Sample):
    return 1000


def elo_init_random(sample: Sample):
    return 1000 + 100 * random.random() - 50


class Tournament:
    def __init__(
        self,
        model: str | Model,
        dataset: Dataset,
        sample_to_binary: Callable[[Sample], bool],
        comparison_prompt: ComparisonPrompt,
        log_dir: str,
        max_players: int | None = None,
        elo_initializer: Callable[[Sample], float] = elo_init_constant,
    ):
        self.model = get_model(model)
        self.dataset = dataset
        self.sample_to_binary = sample_to_binary
        self.comparison_prompt = comparison_prompt
        self.tournament_id = uuid()
        self.log_dir = log_dir
        self.players: dict[str, Player] = {}
        for i, sample in enumerate(dataset[:max_players], 1):
            pid = str(sample.id or i)
            starting_elo = elo_initializer(sample)
            self.players[pid] = Player(pid, sample, starting_elo)

    def metrics(self) -> dict[str, Any]:
        labels: list[int] = []
        scores: list[float] = []
        for p in self.players.values():
            labels.append(int(self.sample_to_binary(p.sample)))
            scores.append(p.expected_win_prob())
        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "acc": get_best_accuracy(labels, scores),
            "f1": float(get_best_f1(labels, scores)),
        }

    def standings(self):
        return {
            pid: p.elo
            for pid, p in sorted(self.players.items(), key=lambda tup: tup[1].elo)
        }

    async def matchup(self, pid1: str, pid2: str, max_rematches=10):
        """Run a single matchup between two players."""
        p1 = self.players[pid1]
        p2 = self.players[pid2]
        prompt = self.comparison_prompt.get_prompt(p1.sample, p2.sample)

        winner = None
        p1_elo_before = p1.elo
        p2_elo_before = p2.elo

        model_output = None
        rematches = 0
        for rematches in range(max_rematches):
            model_output = await self.model.generate(prompt)
            winner = self.comparison_prompt.extract_winner(model_output.completion)
            if winner is not None:
                Player.update_elos(p1, p2, winner == "p1")
                return MatchResult(
                    pid1=pid1,
                    pid2=pid2,
                    winner=pid1 if winner == "p1" else pid2,
                    p1_elo_before=p1_elo_before,
                    p1_elo_after=p1.elo,
                    p2_elo_before=p2_elo_before,
                    p2_elo_after=p2.elo,
                    rematches=rematches,
                    model_input=prompt,
                    model_output=model_output,
                )
        else:
            return MatchResult(
                pid1=pid1,
                pid2=pid2,
                winner=None,
                p1_elo_before=p1_elo_before,
                p1_elo_after=p1.elo,
                p2_elo_before=p2_elo_before,
                p2_elo_after=p2.elo,
                rematches=rematches,
                model_input=prompt,
                model_output=model_output,
            )

    async def run_async(self, rounds: int, scheduler: Scheduler):
        """Run the full tournament asynchronously."""

        with TournamentLog(
            log_dir=self.log_dir,
            tournament_id=self.tournament_id,
            model=str(self.model),
            dataset_path=self.dataset.location or "unknown",
            samples=len(self.players),
            n_rounds=rounds,
            scheduler=scheduler.__class__.__name__,
            prompt=self.comparison_prompt,
            start_standings=self.standings(),
            start_metrics=self.metrics(),
        ) as log:
            with TournamentDisplay(log) as disp:
                for round in range(1, rounds + 1):
                    matchups = [
                        (p1.player_id, p2.player_id)
                        for (p1, p2) in scheduler.get_matches(self.players.values())
                    ]

                    log.log_round_start(matchups=matchups)

                    async def run_matchup(pid1: str, pid2: str):
                        result = await self.matchup(pid1, pid2)
                        log.log_match(result)
                        disp.update()

                    await collect(*[run_matchup(*pair) for pair in matchups])

                    log.log_round_end(self.standings(), self.metrics())

    def run(self, rounds: int, scheduler: Scheduler):
        anyio.run(self.run_async, rounds, scheduler)
