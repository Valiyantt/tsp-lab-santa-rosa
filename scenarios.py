"""
scenarios.py - Use case scenarios that compare the algorithms.

Every algorithm gets the SAME wall-clock budget in each scenario (equal-time
comparison), so differences come from the algorithm, not from how long it ran.

  S1  Small daily route      : 10 Santa Rosa stops, tiny budget (50 ms)
  S2  Large overnight batch  : 150 stops, generous budget (6 s)
  S3  Real-time dispatch     : 60 and 150 stops, strict budget (250 ms)
  S4  Orders arrive/cancel   : stops added and removed, budget per change (400 ms)
  S5  Traffic and flooding   : same stops, costs change, budget per change (150 ms)

  python scenarios.py            -> full run (~2 min), writes output/use_case_*
  python scenarios.py --quick    -> shorter budgets, smoke test

Algorithms: Genetic Algorithm (ga.py), Simulated Annealing, Tabu Search,
plus Nearest Neighbour and 2-opt as instant references. 'warm' means the
algorithm continues from its previous tour (repaired after a change);
'restart' means it starts again from scratch.
"""
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga import GeneticTSP, build_matrix, tour_length
from baselines import nearest_neighbour, two_opt, held_karp
import santa_rosa as sr

QUICK = "--quick" in sys.argv
Q = 0.25 if QUICK else 1.0
OUT = "output"
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "use_case_results.txt"), "w")
now = time.perf_counter


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def mean(x):
    return sum(x) / len(x)


# ----------------------------------------------------------------------------
# Time-budgeted versions of the algorithms
# ----------------------------------------------------------------------------
def sa_timed(D, start, budget, seed=0, t0_scale=1.0):
    """Simulated annealing; temperature falls geometrically over the budget."""
    rng = random.Random(seed)
    n = len(D)
    tour = start[:]
    cur = tour_length(tour, D)
    best, bl = tour[:], cur
    T0 = t0_scale * cur / n
    T = T0
    t_start, it = now(), 0
    while True:
        if it % 200 == 0:
            frac = (now() - t_start) / budget
            if frac >= 1:
                break
            T = max(T0 * (1e-3 ** frac), 1e-12)
        it += 1
        i, j = sorted(rng.sample(range(n), 2))
        if i == 0 and j == n - 1:
            continue
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:j + 1] = reversed(tour[i:j + 1])
            cur += delta
            if cur < bl - 1e-12:
                bl, best = cur, tour[:]
    return best, tour_length(best, D)


def tabu_timed(D, start, budget, seed=0, tenure=15, sample=100):
    """Tabu search over sampled 2-opt moves with edge tabu list + aspiration."""
    rng = random.Random(seed)
    n = len(D)
    tour = start[:]
    cur = tour_length(tour, D)
    best, bl = tour[:], cur
    tabu, it, t_start = {}, 0, now()
    while now() - t_start < budget:
        move, md = None, float("inf")
        for _ in range(sample):
            i, j = sorted(rng.sample(range(n), 2))
            if i == 0 and j == n - 1:
                continue
            a, b = tour[i - 1], tour[i]
            c, d = tour[j], tour[(j + 1) % n]
            delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
            key = (min(a, c), max(a, c))
            if tabu.get(key, -1) > it and cur + delta >= bl:
                continue
            if delta < md:
                move, md = (i, j, key), delta
        it += 1
        if move is None:
            continue
        i, j, key = move
        tour[i:j + 1] = reversed(tour[i:j + 1])
        cur += md
        tabu[key] = it + tenure
        if cur < bl - 1e-12:
            bl, best = cur, tour[:]
    return best, tour_length(best, D)


def run_ga(ga, budget, t_start=None):
    """Evolve until the budget is used (always at least one generation)."""
    t0 = t_start if t_start is not None else now()
    while True:
        ga.step()
        if now() - t0 >= budget:
            break
    return ga.best()[1]


def insert_city(tour, D, new):
    bp, bc = 0, float("inf")
    for p in range(len(tour)):
        a, b = tour[p - 1], tour[p]
        c = D[a][new] + D[new][b] - D[a][b]
        if c < bc:
            bc, bp = c, p
    t = tour[:]
    t.insert(bp, new)
    return t


def remove_city(tour, idx):
    return [c - 1 if c > idx else c for c in tour if c != idx]


def rand_points(n, rng):
    return [(rng.random() * 100, rng.random() * 100) for _ in range(n)]


RESULTS = {}


# ----------------------------------------------------------------------------
# S1: small daily route (Santa Rosa, 10 stops)
# ----------------------------------------------------------------------------
def s1():
    B = 0.05 * Q
    sc = sr.Scenario()
    D = sc.cost_matrix()
    opt = held_karp(D)
    nn = nearest_neighbour(D)
    names = ["Nearest Neighbour", "2-opt", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
    gaps = {k: [] for k in names}
    for seed in range(5):
        gaps["Nearest Neighbour"].append(tour_length(nn, D))
        gaps["2-opt"].append(tour_length(two_opt(nn, D), D))
        gaps["Simulated Annealing"].append(sa_timed(D, nn, B, seed)[1])
        gaps["Tabu Search"].append(tabu_timed(D, nn, B, seed)[1])
        t0 = now()
        ga = GeneticTSP(sc.points, pop_size=60, seed=seed, dist_fn=sr.road_km)
        ga.set_distances(D)
        gaps["Genetic Algorithm"].append(run_ga(ga, B, t0))
    log("=" * 78)
    log(f"S1 SMALL DAILY ROUTE - 10 Santa Rosa stops, {B * 1000:.0f} ms budget, 5 seeds")
    log(f"   exact optimum (Held-Karp) = {opt:.2f} effective km")
    out = {}
    for k in names:
        g = max(0.0, 100 * (mean(gaps[k]) - opt) / opt)
        out[k] = g
        log(f"   {k:<20} mean cost {mean(gaps[k]):7.2f}   gap to optimum {g:5.2f}%")
    RESULTS["S1"] = out


# ----------------------------------------------------------------------------
# S2: large overnight batch (150 stops)
# ----------------------------------------------------------------------------
def s2():
    B = 6.0 * Q
    n, trials = 150, 2
    names = ["Nearest Neighbour", "2-opt", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
    res = {k: [] for k in names}
    for tr in range(trials):
        pts = rand_points(n, random.Random(500 + tr))
        D = build_matrix(pts)
        nn = nearest_neighbour(D)
        res["Nearest Neighbour"].append(tour_length(nn, D))
        res["2-opt"].append(tour_length(two_opt(nn, D), D))
        res["Simulated Annealing"].append(sa_timed(D, nn, B, tr)[1])
        res["Tabu Search"].append(tabu_timed(D, nn, B, tr)[1])
        t0 = now()
        ga = GeneticTSP(pts, pop_size=80, seed=tr)
        res["Genetic Algorithm"].append(run_ga(ga, B, t0))
    log("=" * 78)
    log(f"S2 LARGE OVERNIGHT BATCH - {n} stops, {B:.1f} s budget, {trials} instances")
    out = {}
    for k in names:
        out[k] = mean(res[k])
        log(f"   {k:<20} mean tour length {out[k]:8.1f}")
    RESULTS["S2"] = out


# ----------------------------------------------------------------------------
# S3: real-time dispatch (strict budget)
# ----------------------------------------------------------------------------
def s3():
    B = 0.25 * Q
    names = ["Nearest Neighbour", "2-opt", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
    log("=" * 78)
    log(f"S3 REAL-TIME DISPATCH - strict {B * 1000:.0f} ms budget, 5 instances each size")
    RESULTS["S3"] = {}
    for n in (60, 150):
        res = {k: [] for k in names}
        gens = []
        for tr in range(5):
            pts = rand_points(n, random.Random(900 + n + tr))
            D = build_matrix(pts)
            nn = nearest_neighbour(D)
            res["Nearest Neighbour"].append(tour_length(nn, D))
            res["2-opt"].append(tour_length(two_opt(nn, D), D))
            res["Simulated Annealing"].append(sa_timed(D, nn, B, tr)[1])
            res["Tabu Search"].append(tabu_timed(D, nn, B, tr)[1])
            t0 = now()
            ga = GeneticTSP(pts, pop_size=80, seed=tr)
            res["Genetic Algorithm"].append(run_ga(ga, B, t0))
            gens.append(ga.generation)
        log(f"   n = {n}  (GA completed about {mean(gens):.0f} generations)")
        RESULTS["S3"][n] = {}
        for k in names:
            RESULTS["S3"][n][k] = mean(res[k])
            log(f"      {k:<20} mean tour length {mean(res[k]):8.1f}")


# ----------------------------------------------------------------------------
# S4: orders arrive and cancel (stops added / removed)
# ----------------------------------------------------------------------------
def s4():
    B = 0.4 * Q
    events, insts = 6, 3
    names = ["GA adaptive", "GA restart", "SA warm", "SA restart", "Tabu warm", "Tabu restart",
             "Nearest Neighbour", "2-opt"]
    above = {k: [] for k in names}
    for inst in range(insts):
        rng = random.Random(2000 + inst)
        pts = rand_points(60, rng)
        D = build_matrix(pts)
        nn = nearest_neighbour(D)
        ga_ad = GeneticTSP(pts, pop_size=80, seed=inst, local_rate=0.05)
        run_ga(ga_ad, B)
        warm_sa, warm_tb = sa_timed(D, nn, B, inst)[0], tabu_timed(D, nn, B, inst)[0]
        for ev in range(events):
            for _ in range(3):                       # 3 new orders
                p = (rng.random() * 100, rng.random() * 100)
                pts.append(p)
                D = build_matrix(pts)
                new = len(pts) - 1
                ga_ad.add_city(p)
                warm_sa = insert_city(warm_sa, D, new)
                warm_tb = insert_city(warm_tb, D, new)
            idx = rng.randrange(1, len(pts))         # 1 cancelled order
            del pts[idx]
            D = build_matrix(pts)
            ga_ad.remove_city(idx)
            warm_sa, warm_tb = remove_city(warm_sa, idx), remove_city(warm_tb, idx)
            nn = nearest_neighbour(D)
            cost = {}
            cost["GA adaptive"] = run_ga(ga_ad, B)
            t0 = now()
            g2 = GeneticTSP(pts, pop_size=80, seed=ev, local_rate=0.05)
            cost["GA restart"] = run_ga(g2, B, t0)
            warm_sa, cost["SA warm"] = sa_timed(D, warm_sa, B, ev, t0_scale=0.1)
            cost["SA restart"] = sa_timed(D, nn, B, ev)[1]
            warm_tb, cost["Tabu warm"] = tabu_timed(D, warm_tb, B, ev)
            cost["Tabu restart"] = tabu_timed(D, nn, B, ev)[1]
            cost["Nearest Neighbour"] = tour_length(nn, D)
            cost["2-opt"] = tour_length(two_opt(nn, D), D)
            bestk = min(cost.values())
            for k in names:
                above[k].append(100 * (cost[k] - bestk) / bestk)
    log("=" * 78)
    log(f"S4 ORDERS ARRIVE AND CANCEL - start 60 stops, {events} changes (+3 / -1 stops), "
        f"{B * 1000:.0f} ms per change, {insts} instances")
    log("   Mean % above the best tour found by any method at that change (0 = best):")
    out = {}
    for k in names:
        out[k] = mean(above[k])
        log(f"   {k:<14} {out[k]:6.2f}%")
    RESULTS["S4"] = out


# ----------------------------------------------------------------------------
# S5: traffic and flooding (same stops, changing costs) - Santa Rosa
# ----------------------------------------------------------------------------
def s5():
    B = 0.15 * Q
    phases = [("normal", False, False), ("traffic", True, False),
              ("flood", True, True), ("cleared", False, False)]
    names = ["Nearest Neighbour", "2-opt", "GA adaptive", "GA restart",
             "SA warm", "SA restart", "Tabu warm", "Tabu restart"]
    gaps = {k: [] for k in names}
    per_phase = {k: {p[0]: [] for p in phases} for k in names}
    for seed in range(5):
        sc = sr.Scenario()
        ga = GeneticTSP(sc.points, pop_size=60, seed=seed, dist_fn=sr.road_km, local_rate=0.05)
        warm_sa = warm_tb = list(range(len(sc.points)))
        for name, traffic, flood in phases:
            sc.traffic, sc.flood = traffic, flood
            D = sc.cost_matrix()
            opt = held_karp(D)
            nn = nearest_neighbour(D)
            cost = {"Nearest Neighbour": tour_length(nn, D),
                    "2-opt": tour_length(two_opt(nn, D), D)}
            ga.set_distances(D)
            cost["GA adaptive"] = run_ga(ga, B)
            t0 = now()
            g2 = GeneticTSP(sc.points, pop_size=60, seed=seed, dist_fn=sr.road_km, local_rate=0.05)
            g2.set_distances(D)
            cost["GA restart"] = run_ga(g2, B, t0)
            warm_sa, cost["SA warm"] = sa_timed(D, warm_sa, B, seed, t0_scale=0.1)
            cost["SA restart"] = sa_timed(D, nn, B, seed)[1]
            warm_tb, cost["Tabu warm"] = tabu_timed(D, warm_tb, B, seed)
            cost["Tabu restart"] = tabu_timed(D, nn, B, seed)[1]
            for k in names:
                g = max(0.0, 100 * (cost[k] - opt) / opt)
                gaps[k].append(g)
                per_phase[k][name].append(g)
    log("=" * 78)
    log(f"S5 TRAFFIC AND FLOODING - 10 Santa Rosa stops, 4 cost changes, "
        f"{B * 1000:.0f} ms per change, 5 seeds")
    log("   Mean % above the exact optimum (Held-Karp), 0 = optimal:")
    out = {}
    for k in names:
        out[k] = mean(gaps[k])
        detail = "  ".join(f"{p[0]} {mean(per_phase[k][p[0]]):5.2f}%" for p in phases)
        log(f"   {k:<18} overall {out[k]:6.2f}%   ({detail})")
    RESULTS["S5"] = out


# ----------------------------------------------------------------------------
def plots():
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    cols = {"Nearest Neighbour": "#999999", "2-opt": "#c9a227", "Simulated Annealing": "#4c72b0",
            "Tabu Search": "#dd8452", "Genetic Algorithm": "#55a868"}

    def bars(a, d, title, ylabel, colmap=None):
        keys = list(d)
        a.bar(range(len(keys)), [d[k] for k in keys],
              color=[(colmap or {}).get(k, "#4c72b0") for k in keys])
        a.set_xticks(range(len(keys)))
        a.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
        a.set_title(title, fontsize=10)
        a.set_ylabel(ylabel)
        a.grid(axis="y", alpha=.3)

    bars(ax[0][0], RESULTS["S1"], "S1 Small daily route", "% above optimum", cols)
    bars(ax[0][1], RESULTS["S2"], "S2 Large batch (150 stops)", "tour length", cols)
    d3 = RESULTS["S3"][150]
    bars(ax[0][2], d3, "S3 Real-time dispatch (150 stops)", "tour length", cols)
    c4 = {k: ("#55a868" if k.startswith("GA") else "#4c72b0" if k.startswith("SA") else "#dd8452" if k.startswith("Tabu") else "#999999") for k in RESULTS["S4"]}
    bars(ax[1][0], RESULTS["S4"], "S4 Orders arrive/cancel", "% above best", c4)
    c5 = {k: ("#55a868" if k.startswith("GA") else "#4c72b0" if k.startswith("SA") else
              "#dd8452" if k.startswith("Tabu") else "#999999") for k in RESULTS["S5"]}
    bars(ax[1][1], RESULTS["S5"], "S5 Traffic and flooding", "% above optimum", c5)
    d3b = RESULTS["S3"][60]
    bars(ax[1][2], d3b, "S3 Real-time dispatch (60 stops)", "tour length", cols)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "use_case_comparison.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    t0 = now()
    for f in (s1, s2, s3, s4, s5):
        f()
    plots()
    log("=" * 78)
    log(f"Done in {now() - t0:.0f} s. Saved output/use_case_comparison.png")
