"""
In-play tennis win probability engine.

Hierarchical Markov model: point -> game -> set -> match.
Assumes points are i.i.d. conditional on who is serving (the standard
Klaassen & Magnus assumption; empirically close enough to be useful, and
the places it breaks are exactly where a trader adds value).

Player A and Player B each have a serve-point-win probability. Everything
else falls out of the chain.

No third-party dependencies.
"""

from functools import lru_cache


# --------------------------------------------------------------------------
# Serve probability estimation
# --------------------------------------------------------------------------

# Tour-level baselines for serve points won. Update these per surface/tour.
TOUR_AVG_SERVE = {
    "atp_hard": 0.645,
    "atp_clay": 0.630,
    "atp_grass": 0.665,
    "wta_hard": 0.585,
    "wta_clay": 0.570,
    "wta_grass": 0.600,
}


def serve_probability(server_spw, returner_rpw, tour_avg):
    """
    Combine a server's serve-points-won rate with the returner's
    return-points-won rate into a matchup-specific point probability.

    p = server_spw - returner_rpw + (1 - tour_avg)

    server_spw   : server's career/recent serve points won (e.g. 0.68)
    returner_rpw : opponent's return points won (e.g. 0.38)
    tour_avg     : baseline serve points won for the tour/surface

    Clamped to a sane range so bad inputs cannot produce nonsense.
    """
    p = server_spw - returner_rpw + (1 - tour_avg)
    return min(max(p, 0.30), 0.90)


# --------------------------------------------------------------------------
# Game level
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def game_win_prob(p, server_points=0, returner_points=0):
    """
    Probability the server wins the game from a given point score.
    Points are indexed 0,1,2,3 = 0,15,30,40. Deuce and advantage are
    handled by collapsing to the standard deuce formula.
    """
    # Deuce or beyond: collapse by point difference.
    if server_points >= 3 and returner_points >= 3:
        diff = server_points - returner_points
        d = deuce_win_prob(p)
        if diff >= 1:
            # Advantage server
            return p + (1 - p) * d
        if diff <= -1:
            # Advantage returner
            return p * d
        return d

    if server_points == 4:
        return 1.0
    if returner_points == 4:
        return 0.0

    win = p * game_win_prob(p, server_points + 1, returner_points)
    lose = (1 - p) * game_win_prob(p, server_points, returner_points + 1)
    return win + lose


@lru_cache(maxsize=None)
def deuce_win_prob(p):
    """Server's probability of winning from deuce: p^2 / (p^2 + (1-p)^2)."""
    return (p * p) / (p * p + (1 - p) * (1 - p))


# --------------------------------------------------------------------------
# Tiebreak level
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def tiebreak_deuce_prob(pa, pb):
    """
    From a tied tiebreak score at or past target-1, probability A wins.
    Over the next pair of points A serves one (wins with pa) and B serves
    the other (A wins with 1-pb). A must take both to win the tiebreak.
    """
    a_both = pa * (1 - pb)
    b_both = (1 - pa) * pb
    if a_both + b_both == 0:
        return 0.5
    return a_both / (a_both + b_both)


@lru_cache(maxsize=None)
def tiebreak_win_prob(pa, pb, a_points=0, b_points=0, a_serving=True, target=7):
    """
    Probability player A wins the tiebreak.

    pa / pb  : serve point win probability for A and B
    a_serving: whether A serves the current point
    target   : 7 for a standard tiebreak, 10 for a match tiebreak
    """
    if a_points >= target and a_points - b_points >= 2:
        return 1.0
    if b_points >= target and b_points - a_points >= 2:
        return 0.0

    # Both at or past target-1: the rest is a win-by-two race. Serves
    # alternate in pairs, so over any two points from an even total each
    # player serves once. Collapse to a closed form instead of recursing
    # forever.
    if a_points >= target - 1 and b_points >= target - 1:
        d = tiebreak_deuce_prob(pa, pb)
        diff = a_points - b_points
        if diff == 0:
            return d
        p = pa if a_serving else (1 - pb)
        if diff == 1:
            return p + (1 - p) * d
        return p * d

    p = pa if a_serving else (1 - pb)

    # Tiebreak serve rotation: first point single serve, then alternating pairs.
    played = a_points + b_points
    next_played = played + 1
    if played == 0:
        switch = True
    else:
        switch = (next_played % 2 == 1)
    next_a_serving = (not a_serving) if switch else a_serving

    win = p * tiebreak_win_prob(pa, pb, a_points + 1, b_points, next_a_serving, target)
    lose = (1 - p) * tiebreak_win_prob(pa, pb, a_points, b_points + 1, next_a_serving, target)
    return win + lose


# --------------------------------------------------------------------------
# Set level
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def set_win_prob(pa, pb, a_games=0, b_games=0, a_serving=True,
                 a_points=0, b_points=0, in_tiebreak=False,
                 tb_a=0, tb_b=0, tb_a_serving=True):
    """
    Probability player A wins the set from an arbitrary in-set state.
    """
    if in_tiebreak:
        tb = tiebreak_win_prob(pa, pb, tb_a, tb_b, tb_a_serving)
        return tb

    # Set already decided
    if a_games >= 6 and a_games - b_games >= 2:
        return 1.0
    if b_games >= 6 and b_games - a_games >= 2:
        return 0.0
    if a_games == 7:
        return 1.0
    if b_games == 7:
        return 0.0

    # 6-6: tiebreak
    if a_games == 6 and b_games == 6:
        return tiebreak_win_prob(pa, pb, 0, 0, a_serving)

    # Resolve the current game
    p_server = pa if a_serving else pb
    if a_serving:
        p_hold = game_win_prob(p_server, a_points, b_points)
        p_a_wins_game = p_hold
    else:
        p_hold = game_win_prob(p_server, b_points, a_points)
        p_a_wins_game = 1 - p_hold

    won = set_win_prob(pa, pb, a_games + 1, b_games, not a_serving)
    lost = set_win_prob(pa, pb, a_games, b_games + 1, not a_serving)
    return p_a_wins_game * won + (1 - p_a_wins_game) * lost


# --------------------------------------------------------------------------
# Match level
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def match_win_prob(pa, pb, a_sets=0, b_sets=0, best_of=3,
                   a_games=0, b_games=0, a_serving=True,
                   a_points=0, b_points=0, in_tiebreak=False,
                   tb_a=0, tb_b=0, tb_a_serving=True):
    """
    Probability player A wins the match from a full in-play state.

    a_sets / b_sets : sets already won
    a_games/b_games : games in the current set
    a_points/b_points: points in the current game (0..4, 3=40, 4+=adv logic)
    a_serving       : is A serving the current game
    in_tiebreak     : currently in a tiebreak
    tb_a / tb_b     : tiebreak points
    """
    need = best_of // 2 + 1
    if a_sets >= need:
        return 1.0
    if b_sets >= need:
        return 0.0

    p_set = set_win_prob(pa, pb, a_games, b_games, a_serving,
                         a_points, b_points, in_tiebreak,
                         tb_a, tb_b, tb_a_serving)

    # Next set starts fresh. Server of the next set alternates from the
    # last game of the current set; approximated by alternating from
    # the current server, which is exact at set start.
    won = match_win_prob(pa, pb, a_sets + 1, b_sets, best_of, 0, 0, not a_serving)
    lost = match_win_prob(pa, pb, a_sets, b_sets + 1, best_of, 0, 0, not a_serving)
    return p_set * won + (1 - p_set) * lost


# --------------------------------------------------------------------------
# Pricing helpers
# --------------------------------------------------------------------------

def to_decimal_odds(prob, margin=0.0):
    """Fair decimal odds, optionally with an applied margin (0.05 = 5%)."""
    if prob <= 0:
        return float("inf")
    return round(1.0 / (prob * (1 + margin)), 3)


def devig_two_way(odds_a, odds_b):
    """Strip overround from a two-way market. Returns implied true probs."""
    ia, ib = 1 / odds_a, 1 / odds_b
    total = ia + ib
    return ia / total, ib / total


def expected_value(model_prob, offered_odds):
    """EV per unit staked at the offered price."""
    return round(model_prob * (offered_odds - 1) - (1 - model_prob), 4)


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tour_avg = TOUR_AVG_SERVE["atp_hard"]

    # Player A: strong server, average returner. Player B: the reverse.
    a_spw, a_rpw = 0.690, 0.360
    b_spw, b_rpw = 0.640, 0.400

    pa = serve_probability(a_spw, b_rpw, tour_avg)
    pb = serve_probability(b_spw, a_rpw, tour_avg)

    print(f"A serve point win: {pa:.4f}")
    print(f"B serve point win: {pb:.4f}")
    print(f"A hold prob:       {game_win_prob(pa):.4f}")
    print(f"B hold prob:       {game_win_prob(pb):.4f}")

    pre = match_win_prob(pa, pb, best_of=3)
    print(f"\nPre-match A win:   {pre:.4f}  (fair {to_decimal_odds(pre)})")

    print("\nIn-play states (best of 3):")
    states = [
        ("Level, 0-0 first set, A serving", dict(a_sets=0, b_sets=0, a_games=0, b_games=0, a_serving=True)),
        ("A broken, 2-4 down set 1, B serving", dict(a_sets=0, b_sets=0, a_games=2, b_games=4, a_serving=False)),
        ("A lost set 1, 3-3 set 2, A serving", dict(a_sets=0, b_sets=1, a_games=3, b_games=3, a_serving=True)),
        ("A won set 1, 5-4 up set 2, A serving 40-30", dict(a_sets=1, b_sets=0, a_games=5, b_games=4, a_serving=True, a_points=3, b_points=2)),
        ("Set 1 tiebreak, A leads 5-3, A serving", dict(a_sets=0, b_sets=0, a_games=6, b_games=6, in_tiebreak=True, tb_a=5, tb_b=3, tb_a_serving=True)),
    ]
    for label, kw in states:
        p = match_win_prob(pa, pb, best_of=3, **kw)
        print(f"  {label:45s} {p:.4f}   fair {to_decimal_odds(p):>7}")

    # Market comparison example
    print("\nMarket check:")
    market_a, market_b = 1.55, 2.55
    true_a, true_b = devig_two_way(market_a, market_b)
    model_a = match_win_prob(pa, pb, best_of=3)
    print(f"  Market implied A (devigged): {true_a:.4f}")
    print(f"  Model A:                     {model_a:.4f}")
    print(f"  EV backing A at {market_a}:      {expected_value(model_a, market_a):+.4f} per unit")
