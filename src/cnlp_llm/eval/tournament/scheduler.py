import itertools
import math
import random
from collections.abc import Iterable
from typing import Protocol

import networkx as nx

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


class OutsideInScheduler(Scheduler):
    def get_matches(self, players: Iterable[Player]):
        players = sorted(players, key=lambda p: p.elo)
        yield from zip(players[: len(players) // 2], reversed(players))


class SlidingScheduler(Scheduler):
    def get_matches(self, players):
        players = sorted(players, key=lambda p: p.elo)
        yield from zip(players, players[math.ceil(len(players) / 2) :])


class SwissScheduler(Scheduler):
    """
    Swiss system (Dutch)

    https://en.wikipedia.org/wiki/Swiss-system_tournament#Dutch_system
    """

    def __init__(self, group_size: int = 8):
        if group_size % 2 == 1:
            raise ValueError("group size must be even")
        self.group_size = group_size

    def get_matches(self, players: Iterable[Player]):
        players = sorted(players, key=lambda p: p.elo)

        # figure out how many leftovers we'll have and randomly match them up
        remainder = len(players) % self.group_size
        leftovers = [
            players.pop(random.randint(0, len(players) - 1)) for _ in range(remainder)
        ]

        while len(leftovers) >= 2:
            yield leftovers.pop(), leftovers.pop()

        for group_idx in range(0, len(players), self.group_size):
            for offset in range(self.group_size // 2):
                p1 = group_idx + offset
                p2 = group_idx + self.group_size // 2
                yield players[p1], players[p2]


class GraphScheduler(Scheduler):
    def __init__(self):
        self.edges: dict[str, list[str]] = {}

    def add_edge(self, p1: str, p2: str):
        if p2 not in self.edges[p1]:
            self.edges[p1].append(p2)
            self.edges[p2].append(p1)

    def get_matches(self, players: Iterable[Player]):
        player_dict = {p.player_id: p for p in players}

        # construct graph if it's the first round
        if len(self.edges) == 0:
            self.edges = {pid: [] for pid in player_dict}

        # use networkx to get shortest distances between each pair of players
        distance_dict: dict[str, dict[str, int]] = dict(  # type: ignore
            nx.all_pairs_shortest_path_length(nx.from_dict_of_lists(self.edges))  # type: ignore
        )

        # get distance for each unique combination of players,
        distance_list: list[tuple[int, str, str]] = []
        for p1, p2 in itertools.combinations(player_dict.keys(), 2):
            if p1 in distance_dict and p2 in distance_dict[p1]:
                distance_list.append((distance_dict[p1][p2], p1, p2))
            else:
                # no path, use len(player_dict) as max value
                distance_list.append((len(player_dict), p1, p2))

        # sort the list, most distant pairs first
        distance_list.sort(reverse=True)

        # keep track of which players have already been assigned a match
        assigned: set[str] = set()
        for dist, p1, p2 in distance_list:
            if p1 in assigned or p2 in assigned:
                continue

            # yield the pair and update the graph
            yield player_dict[p1], player_dict[p2]
            self.add_edge(p1, p2)
            assigned.update([p1, p2])
