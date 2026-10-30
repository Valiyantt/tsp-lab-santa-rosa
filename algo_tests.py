"""
algo_tests.py - Run the TEST CASES and USE CASES against every algorithm.

Algorithms: Nearest Neighbour, 2-opt, Simulated Annealing, Tabu Search, Genetic Algorithm.

PART A  test cases TC-01..TC-07 (one piece of behaviour each, expected vs actual, PASS/FAIL)
PART B  use case acceptance matrix UC-01..UC-05 (can this algorithm serve the use case?)

PASS THRESHOLDS ARE FIXED UP FRONT (my own choices, change them if you disagree):
  exact-optimum gap on 10 stops : GA, SA, Tabu <= 1%   2-opt <= 10%   Nearest Neighbour <= 40%
  use cases UC-02, UC-03, UC-04 : within 2% of the best method
  use cases UC-01, UC-05        : within 1% of the exact optimum

  python algo_tests.py            -> full run (~3 min)
  python algo_tests.py --quick    -> short budgets
"""
import random
import sys

import scenarios as sc
from ga import GeneticTSP, build_matrix, tour_length
from baselines import nearest_neighbour, two_opt, held_karp, simulated_annealing, tabu_search

QUICK = "--quick" in sys.argv
ALGOS = ["Nearest Neighbour", "2-opt", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
SHORT = {"Nearest Neighbour": "NN", "2-opt": "2-opt", "Simulated Annealing": "SA",
         "Tabu Search": "Tabu", "Genetic Algorithm": "GA"}
GAP_TARGET = {"Nearest Neighbour": 40.0, "2-opt": 10.0, "Simulated Annealing": 1.0,
              "Tabu Search": 1.0, "Genetic Algorithm": 1.0}
BUDGET = 0.05 if QUICK else 0.15
out = sc.log


def rand_pts(n, seed):
    r = random.Random(seed)
    return [(r.random() * 100, r.random() * 100) for _ in range(n)]


def solve(algo, pts, D, budget=BUDGET, seed=0):
    """One entry point so every algorithm solves the same matrix the same way."""
    nn = nearest_neighbour(D)
    if algo == "Nearest Neighbour":
        return nn
    if algo == "2-opt":
        return two_opt(nn, D)
    if algo == "Simulated Annealing":
        return sc.sa_timed(D, nn, budget, seed)[0]
    if algo == "Tabu Search":
        return sc.tabu_timed(D, nn, budget, seed)[0]
    g = GeneticTSP(pts, pop_size=60, seed=seed)
    g.set_distances(D)
    sc.run_ga(g, budget)
    return g.best()[0]


def valid(t, n):
    return sorted(t) == list(range(n))


# ------------------------------------------------------------------ test cases
def tc01(a):
    sq = [(0, 0), (0, 1), (1, 1), (1, 0)]
    D = build_matrix(sq)
    L = tour_length(solve(a, sq, D, 0.05), D)
    return abs(L - 4) < 1e-9, f"{L:.3f}"


def tc02(a):
    good = 0
    for n, seed in ((10, 1), (50, 2), (150, 3)):
        pts = rand_pts(n, seed)
        good += valid(solve(a, pts, build_matrix(pts), BUDGET, seed), n)
    return good == 3, f"{good} of 3 valid"


def tc03(a):
    worst = 0.0
    for seed in range(3):
        pts = rand_pts(10, seed)
        D = build_matrix(pts)
        opt = held_karp(D)
        worst = max(worst, 100 * (tour_length(solve(a, pts, D, BUDGET, seed), D) - opt) / opt)
    return worst <= GAP_TARGET[a], f"max gap {worst:.2f}%"


def tc04(a):
    import math
    n = 16
    pts = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    random.Random(3).shuffle(pts)
    D = build_matrix(pts)
    L = tour_length(solve(a, pts, D, BUDGET * 2, 3), D)
    per = n * 2 * math.sin(math.pi / n)
    return abs(L - per) < 1e-6, f"{L:.4f}"


def tc05(a):
    r = random.Random(5)
    pts = [(r.random() * 100, r.random() * 100) for _ in range(30)]
    new_pts = [(r.random() * 100, r.random() * 100) for _ in range(5)]
    D = build_matrix(pts)
    if a == "Genetic Algorithm":
        g = GeneticTSP(pts, pop_size=60, seed=5)
        sc.run_ga(g, BUDGET)
        for p in new_pts:
            g.add_city(p)
            sc.run_ga(g, 0.03)
        g.remove_city(3)
        g.remove_city(0)
        sc.run_ga(g, 0.05)
        t, L = g.best()
        n, D2 = len(g.points), build_matrix(g.points)
    elif a in ("Simulated Annealing", "Tabu Search"):
        t = solve(a, pts, D, BUDGET, 5)
        for p in new_pts:                                  # repair, do not restart
            pts.append(p)
            D = build_matrix(pts)
            t = sc.insert_city(t, D, len(pts) - 1)
        for idx in (3, 0):
            del pts[idx]
            t = sc.remove_city(t, idx)
        D2 = build_matrix(pts)
        fn = sc.sa_timed if a == "Simulated Annealing" else sc.tabu_timed
        t = fn(D2, t, 0.05, 5)[0]
        n, L = len(pts), tour_length(t, D2)
    else:                                                  # NN / 2-opt: solve again from scratch
        pts = pts + new_pts
        del pts[3]
        del pts[0]
        D2 = build_matrix(pts)
        t = solve(a, pts, D2, BUDGET, 5)
        n, L = len(pts), tour_length(t, D2)
    ok = n == 33 and valid(t, n) and abs(L - tour_length(t, D2)) < 1e-9
    return ok, f"{n} stops, valid={valid(t, n)}"


def tc06(a):
    pts = rand_pts(50, 7)
    D = build_matrix(pts)
    if a == "Genetic Algorithm":
        h = GeneticTSP(pts, pop_size=60, seed=7).run(60)
    elif a == "Simulated Annealing":
        h = simulated_annealing(D, iterations=20000, seed=7)[2]
    elif a == "Tabu Search":
        h = tabu_search(D, iterations=80, seed=7)[2]
    else:
        return None, "n/a (no iterations)"
    mono = all(h[i + 1] <= h[i] + 1e-9 for i in range(len(h) - 1))
    return mono, "never worsened" if mono else "worsened"


def tc07(a):
    pts = rand_pts(20, 9)
    D1 = build_matrix(pts)
    D2 = [[d * (3 if i != j and 5 in (i, j) else 1) for j, d in enumerate(row)] for i, row in enumerate(D1)]
    if a == "Genetic Algorithm":
        g = GeneticTSP(pts, pop_size=60, seed=9)
        g.set_distances(D1)
        sc.run_ga(g, BUDGET)
        old = tour_length(g.best()[0], D2)
        g.set_distances(D2)
        sc.run_ga(g, BUDGET)
        t = g.best()[0]
    else:
        t1 = solve(a, pts, D1, BUDGET, 9)
        old = tour_length(t1, D2)
        if a == "Simulated Annealing":
            t = sc.sa_timed(D2, t1, BUDGET, 9, t0_scale=0.1)[0]
        elif a == "Tabu Search":
            t = sc.tabu_timed(D2, t1, BUDGET, 9)[0]
        else:
            t = solve(a, pts, D2, BUDGET, 9)
    L = tour_length(t, D2)
    return valid(t, 20), f"{old:.0f} -> {L:.0f}"


CASES = [
    ("TC-01", "Square: the best tour is the edge, length 4", "4.000", tc01),
    ("TC-02", "Valid tour (every stop once) on 10, 50, 150 stops", "3 of 3 valid", tc02),
    ("TC-03", "Gap to the exact Held-Karp optimum, 3 maps of 10 stops", "gap <= target", tc03),
    ("TC-04", "16 points on a circle: best tour is the perimeter", "6.2429", tc04),
    ("TC-05", "30 stops, add 5, remove 2: valid tour of 33 stops", "33 stops, valid", tc05),
    ("TC-06", "Best-so-far length never gets worse", "never worsened", tc06),
    ("TC-07", "Flood (links to one stop cost 3x): valid tour, new cost", "valid", tc07),
]


def part_a():
    out("=" * 100)
    out("PART A  TEST CASES on all five algorithms (budget %.2f s per solve where an algorithm uses time)" % BUDGET)
    out("Gap targets: GA/SA/Tabu <= 1%, 2-opt <= 10%, Nearest Neighbour <= 40% (fixed before running).")
    out("=" * 100)
    tally = {a: [0, 0] for a in ALGOS}
    for tc, what, expected, fn in CASES:
        out(f"\n{tc}  {what}   [expected: {expected}]")
        for a in ALGOS:
            ok, detail = fn(a)
            tag = "n/a " if ok is None else ("PASS" if ok else "FAIL")
            if ok is not None:
                tally[a][1] += 1
                tally[a][0] += bool(ok)
            out(f"      {SHORT[a]:<5} {tag}  {detail}")
    out("\nSummary (passed / applicable test cases):")
    for a in ALGOS:
        out(f"   {a:<20} {tally[a][0]} of {tally[a][1]}")
    return tally


# -------------------------------------------------------------- use case matrix
def pct_above_best(d):
    m = min(d.values())
    return {k: 100 * (v - m) / m for k, v in d.items()}


def best_variant(d):
    """Collapse 'GA adaptive/GA restart' etc. to one value per algorithm (its better variant)."""
    m = {"Nearest Neighbour": d.get("Nearest Neighbour"), "2-opt": d.get("2-opt"),
         "Simulated Annealing": min(d["SA warm"], d["SA restart"]),
         "Tabu Search": min(d["Tabu warm"], d["Tabu restart"]),
         "Genetic Algorithm": min(d["GA adaptive"], d["GA restart"])}
    return m


def part_b():
    R = sc.RESULTS
    uc = [
        ("UC-01", "Small daily route (10 stops, 50 ms)", "<= 1% above optimum", R["S1"], 1.0),
        ("UC-02", "Overnight batch (150 stops, 6 s)", "<= 2% above best", pct_above_best(R["S2"]), 2.0),
        ("UC-03", "Real-time dispatch (250 ms, worst of 60 and 150 stops)", "<= 2% above best",
         {k: max(pct_above_best(R["S3"][60])[k], pct_above_best(R["S3"][150])[k]) for k in ALGOS}, 2.0),
        ("UC-04", "Orders arrive and cancel (400 ms/change, better of warm/restart)", "<= 2% above best",
         best_variant(R["S4"]), 2.0),
        ("UC-05", "Traffic and flood (150 ms/change, better of warm/restart)", "<= 1% above optimum",
         best_variant(R["S5"]), 1.0),
    ]
    out("\n" + "=" * 100)
    out("PART B  USE CASE ACCEPTANCE: can each algorithm serve the use case?  value = % above reference")
    out("=" * 100)
    out(f"{'Use case':<8}{'Criterion':<22}" + "".join(f"{SHORT[a]:<14}" for a in ALGOS))
    tally = {a: 0 for a in ALGOS}
    for uid, what, crit, d, tol in uc:
        cells = ""
        for a in ALGOS:
            v = d[a]
            ok = v <= tol
            tally[a] += ok
            cells += f"{('PASS' if ok else 'FAIL') + ' ' + format(v, '.1f') + '%':<14}"
        out(f"{uid:<8}{crit:<22}{cells}")
        out(f"        {what}")
    out("\nUse cases accepted (of 5): " + ", ".join(f"{SHORT[a]} {tally[a]}" for a in ALGOS))
    return tally


if __name__ == "__main__":
    part_a()
    sc.Q = 0.3 if QUICK else 1.0
    for f in (sc.s1, sc.s2, sc.s3, sc.s4, sc.s5):
        f()
    part_b()
