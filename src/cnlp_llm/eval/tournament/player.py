from inspect_ai.dataset import Sample


class Player:
    def __init__(self, player_id: str, sample: Sample, starting_elo: float):
        self.player_id = player_id
        self.sample = sample
        self.elo = starting_elo
        self.wins = 0
        self.losses = 0
        self.history: list[tuple[str, bool]] = []

    def expected_win_prob(self, opp_elo: float = 1000):
        return 1.0 / (1.0 + (10 ** ((opp_elo - self.elo) / 400)))

    @classmethod
    def update_elos(
        cls, player_1: "Player", player_2: "Player", p1_wins: bool, k_factor: int = 32
    ):
        def player_update(p: "Player", opp_elo: float, opp_id: str, win: bool):
            if win:
                update = k_factor * (1 - p.expected_win_prob(opp_elo))
            else:
                update = k_factor * (-p.expected_win_prob(opp_elo))

            p.elo += update
            p.history.append((opp_id, win))

        p1_elo = player_1.elo
        p2_elo = player_2.elo
        player_update(player_1, p2_elo, player_2.player_id, p1_wins)
        player_update(player_2, p1_elo, player_1.player_id, not p1_wins)

    def __str__(self):
        return f"Player {self.player_id} (elo={self.elo:.2f})"
