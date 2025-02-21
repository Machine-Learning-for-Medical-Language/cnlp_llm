import asyncio
import datetime
import logging
import os
import random
from typing import Callable

from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import get_model
from rich.progress import Progress
from shortuuid import uuid

from ...console import console
from .metrics import get_best_accuracy, get_best_f1, roc_auc_score
from .player import Player
from .prompt import ComparisonPrompt
from .scheduler import Scheduler


def elo_init_constant(sample: Sample):
    return 1000


def elo_init_random(sample: Sample):
    return 1000 + 100 * random.random() - 50


def tournament_logger(
    log_dir: str | None, tournament_id: str, log_level: int = logging.INFO
):
    # follow inspect's scheme for naming log files
    def clean(s: str):
        return s.replace("_", "-").replace("/", "-").replace(":", "-")

    now = datetime.datetime.now().isoformat()
    logger_id = f"{clean(now)}_{clean(tournament_id)}"
    logger = logging.getLogger(logger_id)
    logger.setLevel(log_level)

    if log_dir is None:
        logger.addHandler(logging.NullHandler())
        return logger

    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError(f"{log_dir} does not exist or is not a valid directory")

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_id}.log"))
    logger.addHandler(file_handler)
    return logger


class Tournament:
    def __init__(
        self,
        model: str,
        dataset: Dataset,
        sample_to_binary: Callable[[Sample], bool],
        comparison_prompt: ComparisonPrompt,
        max_players: int | None = None,
        elo_initializer: Callable[[Sample], float] = elo_init_constant,
        log_dir: str | None = None,
        log_level: int = logging.INFO,
    ):
        self.model = get_model(model)
        self.dataset = dataset
        self.sample_to_binary = sample_to_binary
        self.comparison_prompt = comparison_prompt
        self.tournament_id = uuid()
        self.logger = tournament_logger(log_dir, self.tournament_id, log_level)

        self.logger.debug("Initializing players from dataset")
        self.players: dict[str, Player] = {}
        for i, sample in enumerate(dataset[:max_players], 1):
            pid = str(sample.id or i)
            starting_elo = elo_initializer(sample)
            self.players[pid] = Player(pid, sample, starting_elo)

        self.logger.debug(f"{len(self.players)} players initialized")

    def metrics(self):
        labels: list[int] = []
        scores: list[float] = []
        for p in self.players.values():
            labels.append(int(self.sample_to_binary(p.sample)))
            scores.append(p.expected_win_prob())
        return {
            "auroc": roc_auc_score(labels, scores),
            "acc": get_best_accuracy(labels, scores),
            "f1": get_best_f1(labels, scores),
        }

    def standings(self):
        return sorted([p for p in self.players.values()], key=lambda p: p.elo)

    async def matchup(self, pid1: str, pid2: str, max_rematches=10):
        """Run a single matchup between two players."""
        p1 = self.players[pid1]
        p2 = self.players[pid2]
        prompt = self.comparison_prompt.get_prompt(p1.sample, p2.sample)

        winner = None

        for rematches in range(max_rematches):
            model_output = await self.model.generate(prompt)
            winner = self.comparison_prompt.extract_winner(model_output.completion)
            if winner is not None:
                p1_elos, p2_elos = Player.update_elos(p1, p2, winner == "p1")

                if winner == "p1":
                    win_id = pid1
                    win_elo_before, win_elo_after = p1_elos
                    los_id = pid2
                    los_elo_before, los_elo_after = p2_elos
                else:
                    win_id = pid2
                    win_elo_before, win_elo_after = p2_elos
                    los_id = pid1
                    los_elo_before, los_elo_after = p1_elos

                self.logger.debug(
                    f"Player {win_id} ({win_elo_before:.1f} -> {win_elo_after:.1f}) beat player {los_id} ({los_elo_before:.1f} -> {los_elo_after:.1f})"
                    + ("" if rematches == 0 else f" after {rematches} tie(s)")
                )
                break
        else:
            self.logger.debug(f"Player {pid1} tied player {pid2}")
            pass

    async def run_async(self, rounds: int, scheduler: Scheduler):
        """Run the full tournament asynchronously."""

        best_metrics = self.metrics()
        auroc, acc, f1 = best_metrics["auroc"], best_metrics["acc"], best_metrics["f1"]
        console.print(f"[b]Baseline:[/b] {auroc=:.3f} {acc=:.3f} {f1=:.3f}")
        self.logger.info(
            f"Starting tournament: n_players={len(self.players)} rounds={rounds} scheduler={scheduler.__class__.__name__}"
        )
        metrics_str = {k: str(v) for k, v in best_metrics.items()}
        self.logger.info(f"Metrics at baseline: {metrics_str}")
        with Progress() as progress:
            for round in range(1, rounds + 1):
                matchups = list(scheduler.get_matches(self.players.values()))
                round_pbar = progress.add_task(
                    f"Round {round}/{rounds}", total=len(matchups)
                )

                async def run_matchup(pid1: str, pid2: str):
                    await self.matchup(pid1, pid2)
                    progress.advance(round_pbar)

                await asyncio.gather(
                    *[run_matchup(p1.player_id, p2.player_id) for p1, p2 in matchups]
                )

                progress.remove_task(round_pbar)
                progress.refresh()

                metrics = self.metrics()
                auroc, acc, f1 = metrics["auroc"], metrics["acc"], metrics["f1"]
                standings = self.standings()
                winner = str(standings[-1])
                loser = str(standings[0])
                console.print(
                    f"[b]Round {str(round).rjust(len(str(rounds)))} results:[/b] {auroc=:.3f} {acc=:.3f} {f1=:.3f} {winner=} {loser=}"
                )

                new_best = auroc > best_metrics["auroc"]
                if new_best:
                    best_metrics = metrics

                metrics_str = {k: str(v) for k, v in metrics.items()}
                self.logger.info(
                    f"Round {round} complete: {metrics_str}"
                    + (" *new best auroc*" if new_best else "")
                )
                self.logger.info(f"Current winner: {winner}")
                self.logger.info(f"Current loser: {loser}")
                self.logger.debug(
                    f"Full standings: {[str(p) for p in self.standings()]}"
                )

        self.logger.info("Tournament complete")
        self.logger.info(f"Final standings: {[str(p) for p in self.standings()]}")

    def run(self, rounds: int, scheduler: Scheduler):
        asyncio.run(self.run_async(rounds=rounds, scheduler=scheduler))
