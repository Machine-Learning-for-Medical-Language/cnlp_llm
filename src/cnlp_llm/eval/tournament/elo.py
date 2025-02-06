def expected_win_prob(elo: float, opp_elo: float):
    return 1.0 / (1.0 + (10 ** ((opp_elo - elo) / 400)))
