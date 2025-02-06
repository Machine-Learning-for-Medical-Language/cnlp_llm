import asyncio
import random
from typing import Callable

from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import get_model
from rich.progress import Progress

from ...console import console
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
        model: str,
        dataset: Dataset,
        sample_to_binary: Callable[[Sample], bool],
        comparison_prompt: ComparisonPrompt,
        max_players: int | None = None,
        elo_initializer: Callable[[Sample], float] = elo_init_constant,
    ):
        self.model = get_model(model)
        self.dataset = dataset
        self.sample_to_binary = sample_to_binary
        self.comparison_prompt = comparison_prompt

        self.players: dict[str, Player] = {}
        for i, sample in enumerate(dataset[:max_players], 1):
            pid = str(sample.id or i)
            starting_elo = elo_initializer(sample)
            self.players[pid] = Player(pid, sample, starting_elo)

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

        for _ in range(max_rematches):
            model_output = await self.model.generate(prompt)
            winner = self.comparison_prompt.extract_winner(model_output.completion)
            if winner is not None:
                Player.update_elos(p1, p2, winner == "p1")
                break
        else:
            # Tie!
            pass

    async def run_async(self, rounds: int, scheduler: Scheduler):
        """Run the full tournament asynchronously."""

        best_metrics = self.metrics()
        auroc, acc, f1 = best_metrics["auroc"], best_metrics["acc"], best_metrics["f1"]
        console.print(f"[b]Baseline:[/b] {auroc=:.3f} {acc=:.3f} {f1=:.3f}")

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

                if auroc > best_metrics["auroc"]:
                    best_metrics = metrics

    def run(self, rounds: int, scheduler: Scheduler):
        asyncio.run(self.run_async(rounds=rounds, scheduler=scheduler))
