"""
Match Charting Project backtest for the in-play tennis model.

Reads Jeff Sackmann's Match Charting Project point-by-point data, estimates
each player's serve and return strength from their other charted matches on
the same surface, then replays a chosen match point by point and records the
model's match win probability before every point.

Two model variants are run side by side:

  prior     serve probabilities fixed at the pre-match estimate
  in-play   serve probabilities updated after every point, shrinking the
            observed rate toward the prior with a fixed pseudo-count

The gap between them is the interesting part. It is a crude version of the
question a trader answers all night: is what I am watching signal about this
match, or noise around the number I started with.

Usage:
    python mcp_backtest.py                       # defaults to RG 2025 final
    python mcp_backtest.py --match-id <id>
    python mcp_backtest.py --list "Alcaraz"      # find match ids
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from urllib.request import urlretrieve

from tennis_model import (
    TOUR_AVG_SERVE,
    serve_probability,
    match_win_prob,
    tiebreak_win_prob,
    to_decimal_odds,
)

BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
MATCHES_FILE = "charting-m-matches.csv"
POINTS_FILES = ["charting-m-points-2020s.csv"]

DEFAULT_MATCH = "20250608-M-Roland_Garros-F-Jannik_Sinner-Carlos_Alcaraz"

# Pseudo-count for the in-play update: how many points of prior evidence the
# pre-match estimate is worth. Higher means the model trusts the prior longer.
PRIOR_WEIGHT = 120

POINT_MAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def ensure(filename):
    """Download a Match Charting Project file if it is not already local."""
    if not os.path.exists(filename):
        print(f"downloading {filename} ...", file=sys.stderr)
        urlretrieve(f"{BASE}/{filename}", filename)
    return filename


def load_matches():
    """match_id -> metadata dict."""
    out = {}
    with open(ensure(MATCHES_FILE), encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            out[row["match_id"]] = {
                "p1": row["Player 1"],
                "p2": row["Player 2"],
                "surface": (row.get("Surface") or "").strip(),
                "best_of": int(row["Best of"]) if row.get("Best of", "").isdigit() else 3,
                "tournament": row.get("Tournament", ""),
                "date": row.get("Date", ""),
                "round": row.get("Round", ""),
            }
    return out


def iter_points():
    for filename in POINTS_FILES:
        with open(ensure(filename), encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                yield row


# --------------------------------------------------------------------------
# Prior estimation
# --------------------------------------------------------------------------

def estimate_priors(matches, target_id, target_surface, exclude_target=True):
    """
    Aggregate serve and return points won for every player on the target
    surface, across all charted matches except the one being replayed.

    Returns (serve_stats, return_stats) keyed by player name, each a
    [won, played] pair.
    """
    serve = defaultdict(lambda: [0, 0])
    ret = defaultdict(lambda: [0, 0])
    target_points = []

    for row in iter_points():
        mid = row["match_id"]
        meta = matches.get(mid)
        if meta is None:
            continue

        if mid == target_id:
            target_points.append(row)
            if exclude_target:
                continue

        if meta["surface"] != target_surface:
            continue

        svr = row.get("Svr")
        winner = row.get("PtWinner")
        if svr not in ("1", "2") or winner not in ("1", "2"):
            continue

        server = meta["p1"] if svr == "1" else meta["p2"]
        returner = meta["p2"] if svr == "1" else meta["p1"]
        server_won = svr == winner

        serve[server][1] += 1
        ret[returner][1] += 1
        if server_won:
            serve[server][0] += 1
        else:
            ret[returner][0] += 1

    target_points.sort(key=lambda r: int(r["Pt"]))
    return serve, ret, target_points


def rate(stats, player, fallback):
    won, played = stats.get(player, [0, 0])
    if played < 200:
        return fallback, played
    return won / played, played


# --------------------------------------------------------------------------
# State parsing
# --------------------------------------------------------------------------

def parse_state(row):
    """
    Turn one Match Charting Project row into the state before that point,
    from player 1's perspective.

    Returns a dict, or None if the row is unparseable.
    """
    try:
        sets = (int(row["Set1"]), int(row["Set2"]))
        games = (int(row["Gm1"]), int(row["Gm2"]))
    except (ValueError, KeyError):
        return None

    svr = row.get("Svr")
    if svr not in ("1", "2"):
        return None
    p1_serving = svr == "1"

    raw = (row.get("Pts") or "").strip()
    if "-" not in raw:
        return None
    left, right = raw.split("-", 1)

    in_tiebreak = games == (6, 6)

    if in_tiebreak:
        try:
            srv_pts, ret_pts = int(left), int(right)
        except ValueError:
            return None
    else:
        if left not in POINT_MAP or right not in POINT_MAP:
            return None
        srv_pts, ret_pts = POINT_MAP[left], POINT_MAP[right]

    # Pts is always written server-first; convert to player 1 / player 2.
    p1_pts, p2_pts = (srv_pts, ret_pts) if p1_serving else (ret_pts, srv_pts)

    return {
        "sets": sets,
        "games": games,
        "p1_pts": p1_pts,
        "p2_pts": p2_pts,
        "p1_serving": p1_serving,
        "in_tiebreak": in_tiebreak,
    }


def win_prob(state, pa, pb, best_of, final_set_tb_target=10):
    """Player 1's match win probability from a parsed state."""
    a_sets, b_sets = state["sets"]
    need = best_of // 2 + 1
    deciding = a_sets == need - 1 and b_sets == need - 1

    if state["in_tiebreak"]:
        target = final_set_tb_target if deciding else 7
        tb = tiebreak_win_prob(
            pa, pb,
            state["p1_pts"], state["p2_pts"],
            state["p1_serving"], target,
        )
        if deciding:
            return tb
        # Winning this tiebreak takes the set; hand the rest back to the chain.
        won = match_win_prob(pa, pb, a_sets + 1, b_sets, best_of, 0, 0,
                             not state["p1_serving"])
        lost = match_win_prob(pa, pb, a_sets, b_sets + 1, best_of, 0, 0,
                              not state["p1_serving"])
        return tb * won + (1 - tb) * lost

    return match_win_prob(
        pa, pb,
        a_sets, b_sets, best_of,
        state["games"][0], state["games"][1],
        state["p1_serving"],
        state["p1_pts"], state["p2_pts"],
    )


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

def run(match_id, out_csv="win_prob_curve.csv", out_png="win_prob_curve.png"):
    matches = load_matches()
    meta = matches.get(match_id)
    if meta is None:
        sys.exit(f"match_id not found: {match_id}")

    surface = meta["surface"] or "Hard"
    p1, p2 = meta["p1"], meta["p2"]
    best_of = meta["best_of"]

    serve_stats, return_stats, points = estimate_priors(matches, match_id, surface)
    if not points:
        sys.exit(f"no point data for {match_id}")

    key = f"atp_{surface.lower()}"
    tour_avg = TOUR_AVG_SERVE.get(key, TOUR_AVG_SERVE["atp_hard"])

    p1_spw, p1_sn = rate(serve_stats, p1, tour_avg)
    p2_spw, p2_sn = rate(serve_stats, p2, tour_avg)
    p1_rpw, p1_rn = rate(return_stats, p1, 1 - tour_avg)
    p2_rpw, p2_rn = rate(return_stats, p2, 1 - tour_avg)

    pa = serve_probability(p1_spw, p2_rpw, tour_avg)
    pb = serve_probability(p2_spw, p1_rpw, tour_avg)

    print(f"{meta['date']} {meta['tournament']} {meta['round']} ({surface}, best of {best_of})")
    print(f"  {p1}: serve {p1_spw:.3f} ({p1_sn} pts), return {p1_rpw:.3f} ({p1_rn} pts)")
    print(f"  {p2}: serve {p2_spw:.3f} ({p2_sn} pts), return {p2_rpw:.3f} ({p2_rn} pts)")
    print(f"  matchup serve probs: {p1} {pa:.4f} | {p2} {pb:.4f}")

    # Running counts for the in-play variant.
    live = {
        "p1_serve": [p1_spw * PRIOR_WEIGHT, PRIOR_WEIGHT],
        "p2_serve": [p2_spw * PRIOR_WEIGHT, PRIOR_WEIGHT],
        "p1_ret": [p1_rpw * PRIOR_WEIGHT, PRIOR_WEIGHT],
        "p2_ret": [p2_rpw * PRIOR_WEIGHT, PRIOR_WEIGHT],
    }

    rows = []
    for row in points:
        state = parse_state(row)
        if state is None:
            continue

        prior_prob = win_prob(state, pa, pb, best_of)

        live_pa = serve_probability(
            live["p1_serve"][0] / live["p1_serve"][1],
            live["p2_ret"][0] / live["p2_ret"][1],
            tour_avg,
        )
        live_pb = serve_probability(
            live["p2_serve"][0] / live["p2_serve"][1],
            live["p1_ret"][0] / live["p1_ret"][1],
            tour_avg,
        )
        live_prob = win_prob(state, round(live_pa, 4), round(live_pb, 4), best_of)

        rows.append({
            "pt": int(row["Pt"]),
            "score": f"{state['sets'][0]}-{state['sets'][1]} "
                     f"{state['games'][0]}-{state['games'][1]} "
                     f"{row.get('Pts', '')}",
            "server": p1 if state["p1_serving"] else p2,
            "prior_prob": round(prior_prob, 5),
            "live_prob": round(live_prob, 5),
            "prior_odds": to_decimal_odds(prior_prob),
            "live_odds": to_decimal_odds(live_prob),
            "gap": round(live_prob - prior_prob, 5),
        })

        # Update the running estimates with the point just played.
        winner = row.get("PtWinner")
        if winner in ("1", "2"):
            server_won = winner == row["Svr"]
            if state["p1_serving"]:
                live["p1_serve"][0] += 1 if server_won else 0
                live["p1_serve"][1] += 1
                live["p2_ret"][0] += 0 if server_won else 1
                live["p2_ret"][1] += 1
            else:
                live["p2_serve"][0] += 1 if server_won else 0
                live["p2_serve"][1] += 1
                live["p1_ret"][0] += 0 if server_won else 1
                live["p1_ret"][1] += 1

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    peak = max(rows, key=lambda r: r["prior_prob"])
    trough = min(rows, key=lambda r: r["prior_prob"])
    widest = max(rows, key=lambda r: abs(r["gap"]))
    print(f"\n  {len(rows)} points priced")
    print(f"  {p1} peak   {peak['prior_prob']:.4f} at {peak['score']} (pt {peak['pt']})")
    print(f"  {p1} trough {trough['prior_prob']:.4f} at {trough['score']} (pt {trough['pt']})")
    print(f"  widest prior/in-play gap {widest['gap']:+.4f} at {widest['score']} (pt {widest['pt']})")

    plot(rows, p1, p2, meta, out_png)
    print(f"\n  wrote {out_csv} and {out_png}")
    return rows


def plot(rows, p1, p2, meta, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["pt"] for r in rows]
    prior = [r["prior_prob"] * 100 for r in rows]
    livep = [r["live_prob"] * 100 for r in rows]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.axhline(50, color="#999999", lw=0.8, ls="--", zorder=1)
    ax.plot(x, prior, lw=1.6, color="#1f3864", label="Pre-match priors (fixed)", zorder=3)
    ax.plot(x, livep, lw=1.2, color="#c0504d", alpha=0.85,
            label=f"Updated in-play (prior weight {PRIOR_WEIGHT} pts)", zorder=2)

    # Mark set boundaries.
    last = None
    for r in rows:
        sets = r["score"].split()[0]
        if last is not None and sets != last:
            ax.axvline(r["pt"], color="#cccccc", lw=0.8, zorder=0)
            ax2.axvline(r["pt"], color="#cccccc", lw=0.8, zorder=0)
        last = sets

    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{p1} match win probability (%)")
    ax.set_title(
        f"{p1} vs {p2} — {meta['tournament']} {meta['round']} {meta['date']}\n"
        f"Hierarchical point/game/set/match Markov model, priced before every point",
        fontsize=12,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)

    gap = [r["gap"] * 100 for r in rows]
    ax2.axhline(0, color="#999999", lw=0.8)
    ax2.fill_between(x, gap, 0, color="#c0504d", alpha=0.35)
    ax2.set_ylabel("In-play minus\nprior (pts %)")
    ax2.set_xlabel("Point number")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)


def find(matches, needle):
    needle = needle.lower()
    hits = [
        (mid, m) for mid, m in matches.items()
        if needle in m["p1"].lower() or needle in m["p2"].lower()
    ]
    hits.sort(key=lambda kv: kv[1]["date"], reverse=True)
    for mid, m in hits[:40]:
        print(f"{m['date']} {m['tournament']:<22} {m['round']:<4} "
              f"{m['p1']} vs {m['p2']}\n    {mid}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", default=DEFAULT_MATCH)
    ap.add_argument("--list", dest="needle")
    ap.add_argument("--out-csv", default="win_prob_curve.csv")
    ap.add_argument("--out-png", default="win_prob_curve.png")
    args = ap.parse_args()

    if args.needle:
        find(load_matches(), args.needle)
    else:
        run(args.match_id, args.out_csv, args.out_png)
