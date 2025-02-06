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


class SwissScheduler(Scheduler):
    def __init__(self, group_size: int = 8):
        self.round = 0
        if group_size % 2 == 1:
            raise ValueError("group size must be even")
        self.group_size = group_size

    def get_matches(self, players: Iterable[Player]):
        players = list(players)
        self.round += 1

        if self.round == 1:
            for i in range(len(players) // 2):
                yield players[i], players[len(players) - i - 1]
        else:
            players.sort(key=lambda p: p.elo)

            # figure out how many leftovers we'll have and randomly match them up
            remainder = len(players) % self.group_size
            leftovers = [
                players.pop(random.randint(0, len(players) - 1))
                for _ in range(remainder)
            ]

            while len(leftovers) >= 2:
                yield leftovers.pop(), leftovers.pop()

            for group_idx in range(0, len(players), self.group_size):
                for offset in range(self.group_size // 2):
                    p1 = group_idx + offset
                    # TODO(ian) for dutch style swiss, this should be `group_idx + self.group_size // 2`
                    # https://en.wikipedia.org/wiki/Swiss-system_tournament#Dutch_system
                    p2 = group_idx + self.group_size - offset - 1
                    yield players[p1], players[p2]
