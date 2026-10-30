"""
simulate.py - Experiments and live simulation for the GA-TSP project.

  python simulate.py            -> runs everything, writes files to ./output
  python simulate.py --quick    -> smaller/faster settings for a smoke test

Experiment 1 (static)  : compare Nearest Neighbour, 2-opt, Simulated Annealing,
                         Tabu Search and the Genetic Algorithm on random instances.
Experiment 2 (dynamic) : cities are added/removed over time. Compare
                         ADAPTIVE GA (repairs its population) vs RESTART GA
                         (throws everything away on each change).
Output                 : PNG plots, an animated GIF, and results.txt (run log).
"""
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ga import GeneticTSP, build_matrix, tour_length
from baselines import (nearest_neighbour, two_opt, simulated_annealing,
                       tabu_search)

QUICK = "--quick" in sys.argv
OUT = "output"
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "results.txt"), "w")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def rand_points(n, rng):
    return [(rng.random() * 100, rng.random() * 100) for _ in range(n)]


# ----------------------------------------------------------------------------
# Experiment 1: static comparison
# ----------------------------------------------------------------------------
def experiment_static():
    log("=" * 64)
    log("EXPERIMENT 1: static comparison (mean over instances)")
    log("=" * 64)
    sizes = [30, 60] if QUICK else [30, 60, 100]
    trials = 1 if QUICK else 3
    gens = 150 if QUICK else 400
    names = ["NearestNeighbour", "2-opt(NN)", "SimAnnealing", "TabuSearch", "GeneticAlg"]
    table = {n: {s: [] for s in sizes} for n in names}
    times = {n: {s: [] for s in sizes} for n in names}
    ga_curve = {}

    for n in sizes:
        for trial in range(trials):
            rng = random.Random(1000 * n + trial)
            pts = rand_points(n, rng)
            D = build_matrix(pts)

            def record(name, length, t0):
                table[name][n].append(length)
                times[name][n].append(time.time() - t0)

            t0 = time.time(); t = nearest_neighbour(D)
            record("NearestNeighbour", tour_length(t, D), t0)
            t0 = time.time(); t = two_opt(nearest_neighbour(D), D)
            record("2-opt(NN)", tour_length(t, D), t0)
            t0 = time.time()
            _, L, _ = simulated_annealing(D, iterations=60000 if QUICK else 150000, seed=trial)
            record("SimAnnealing", L, t0)
            t0 = time.time()
            _, L, _ = tabu_search(D, iterations=100 if QUICK else 250, seed=trial)
            record("TabuSearch", L, t0)
            t0 = time.time()
            ga = GeneticTSP(pts, pop_size=80, seed=trial)
            hist = ga.run(gens)
            record("GeneticAlg", ga.best()[1], t0)
            if trial == 0:
                ga_curve[n] = hist

    log(f"{'Algorithm':<18}" + "".join(f"n={s:<6}(len / sec)   " for s in sizes))
    for name in names:
        row = f"{name:<18}"
        for s in sizes:
            L = sum(table[name][s]) / len(table[name][s])
            T = sum(times[name][s]) / len(times[name][s])
            row += f"{L:8.1f} / {T:5.2f}s     "
        log(row)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, h in ga_curve.items():
        ax[0].plot(h, label=f"n={n}")
    ax[0].set(title="GA convergence (best tour length per generation)",
              xlabel="Generation", ylabel="Best length")
    ax[0].legend(); ax[0].grid(alpha=.3)
    w = 0.16
    for k, name in enumerate(names):
        vals = [sum(table[name][s]) / len(table[name][s]) for s in sizes]
        ax[1].bar([i + k * w for i in range(len(sizes))], vals, w, label=name)
    ax[1].set_xticks([i + 2 * w for i in range(len(sizes))])
    ax[1].set_xticklabels([f"n={s}" for s in sizes])
    ax[1].set(title="Final tour length (lower is better)", ylabel="Length")
    ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "exp1_static_comparison.png"), dpi=130)
    plt.close()


# ----------------------------------------------------------------------------
# Experiment 2: dynamic environment
# ----------------------------------------------------------------------------
def make_event_schedule(total_gens, period, seed):
    """Every `period` generations: add 3 cities, remove 1 (a delivery 'cancelled')."""
    rng = random.Random(seed)
    events = {}
    for g in range(period, total_gens, period):
        events[g] = {"add": [(rng.random() * 100, rng.random() * 100) for _ in range(3)],
                     "remove_rand": rng.random()}
    return events


def experiment_dynamic():
    log("")
    log("=" * 64)
    log("EXPERIMENT 2: dynamic cities, adaptive GA vs restart GA")
    log("=" * 64)
    total = 240 if QUICK else 600
    period = 60
    start_n = 120
    pts0 = rand_points(start_n, random.Random(11))
    events = make_event_schedule(total, period, 12)

    LR = 0.03   # light local search so progress is visible over time
    adaptive = GeneticTSP(pts0, pop_size=80, seed=1, local_rate=LR)
    restart = GeneticTSP(pts0, pop_size=80, seed=1, local_rate=LR)
    ref_len = []                      # 2-opt(NN) reference on the current map
    hist_a, hist_r, n_hist, frames = [], [], [], []
    event_gens = []

    for g in range(total):
        if g in events:
            ev = events[g]
            for p in ev["add"]:
                adaptive.add_city(p)
            idx = int(ev["remove_rand"] * len(adaptive.points))
            adaptive.remove_city(idx)
            # Restart GA: same new map, but population thrown away
            new_pts = list(adaptive.points)
            restart = GeneticTSP(new_pts, pop_size=80, seed=g, greedy_seed=False, local_rate=LR)
            event_gens.append(g)
            log(f"gen {g:4d}: +3 cities, -1 city -> now {len(new_pts)} cities")
        adaptive.step(); restart.step()
        ha, hr = adaptive.best()[1], restart.best()[1]
        hist_a.append(ha); hist_r.append(hr); n_hist.append(len(adaptive.points))
        if g % 5 == 0:
            frames.append((g, list(adaptive.points), adaptive.best()[0], ha,
                           g in events))

    # log recovery right after each change
    log("")
    log("Best tour length k generations after each change (lower = recovered faster):")
    log(f"{'gen':>5} {'k':>3} {'adaptive':>10} {'restart':>10}")
    for g in event_gens:
        for k in (1, 10, 40):
            kk = min(g + k, total - 1)
            log(f"{g:5d} {k:3d} {hist_a[kk]:10.1f} {hist_r[kk]:10.1f}")
    log(f"\nFinal: adaptive={hist_a[-1]:.1f}  restart={hist_r[-1]:.1f}  "
        f"cities={n_hist[-1]}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist_a, label="Adaptive GA (repairs population)")
    ax.plot(hist_r, label="Restart GA (from scratch)", alpha=.8)
    for g in event_gens:
        ax.axvline(g, color="gray", ls=":", lw=.8)
    ax.set(title="Dynamic TSP: cities added/removed every 60 generations",
           xlabel="Generation", ylabel="Best tour length")
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "exp2_dynamic_adaptation.png"), dpi=130)
    plt.close()

    # ---- animation ----
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(i):
        ax.clear()
        g, pts, tour, length, changed = frames[i]
        xs = [pts[c][0] for c in tour] + [pts[tour[0]][0]]
        ys = [pts[c][1] for c in tour] + [pts[tour[0]][1]]
        ax.plot(xs, ys, "-", color="tab:blue", lw=1.3)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], c="tab:red", s=22, zorder=3)
        ax.set(xlim=(-5, 105), ylim=(-5, 105), xticks=[], yticks=[],
               title=f"Gen {g} | cities {len(pts)} | length {length:.0f}"
                     + ("  <- map changed" if changed else ""))

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=120)
    anim.save(os.path.join(OUT, "dynamic_simulation.gif"), writer=PillowWriter(fps=8))
    plt.close()


if __name__ == "__main__":
    experiment_static()
    experiment_dynamic()
    log("\nDone. Files written to ./output")
