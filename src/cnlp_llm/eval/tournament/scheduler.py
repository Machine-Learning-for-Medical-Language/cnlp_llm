import random
from collections.abc import Iterable
from typing import Protocol

from .player import Player


class Scheduler(Protocol):
    def get_matches(
        self, players: Iterable[Player]
    ) -> Iterable[tuple[Player, Player]]: ...


class RandomScheduler(Scheduler):
    def get_matches(self, players: Iterable[Player]):
        players = list(players)
        random.shuffle(players)
        while len(players) > 1:
            yield players.pop(), players.pop()
