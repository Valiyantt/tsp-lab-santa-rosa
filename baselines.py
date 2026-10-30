"""
baselines.py - Comparison algorithms for the research phase.

nearest_neighbour : O(n^2) greedy construction (quality baseline)
two_opt           : O(n^2) per pass local search
simulated_annealing : probabilistic hill climbing with a cooling schedule
tabu_search       : 2-opt neighbourhood with a short-term memory
held_karp         : exact dynamic programming, O(n^2 * 2^n) - small n only,
                    used by the tests as ground truth.
"""
import math
import random
from ga import tour_length


def nearest_neighbour(D, start=0):
    n = len(D)
    left = set(range(n))
    left.remove(start)
    tour = [start]
    while left:
        last = tour[-1]
        nxt = min(left, key=lambda c: D[last][c])
        left.remove(nxt)
        tour.append(nxt)
    return tour


def two_opt(tour, D):
    tour = tour[:]
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                if D[a][c] + D[b][d] < D[a][b] + D[c][d] - 1e-12:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour


def simulated_annealing(D, iterations=200000, t0=None, alpha=None, seed=None):
    """Geometric cooling; neighbour = random segment reversal (2-opt move)."""
    rng = random.Random(seed)
    n = len(D)
    tour = nearest_neighbour(D)
    cur = tour_length(tour, D)
    best, best_len = tour[:], cur
    t0 = t0 or cur / n
    t_end = t0 * 1e-4
    alpha = alpha or (t_end / t0) ** (1.0 / iterations)
    T = t0
    history = []
    for it in range(iterations):
        i, j = sorted(rng.sample(range(n), 2))
        if i == 0 and j == n - 1:
            continue
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour[i:j + 1] = reversed(tour[i:j + 1])
            cur += delta
            if cur < best_len - 1e-12:
                best, best_len = tour[:], cur
        T *= alpha
        if it % 1000 == 0:
            history.append(best_len)
    return best, tour_length(best, D), history


def tabu_search(D, iterations=300, tenure=15, seed=None):
    """Best-improvement over sampled 2-opt moves, with a tabu list of edges."""
    rng = random.Random(seed)
    n = len(D)
    tour = nearest_neighbour(D)
    best, best_len = tour[:], tour_length(tour, D)
    cur_len = best_len
    tabu = {}
    history = []
    for it in range(iterations):
        move, move_delta = None, float("inf")
        for _ in range(min(200, n * n)):
            i, j = sorted(rng.sample(range(n), 2))
            if i == 0 and j == n - 1:
                continue
            a, b = tour[i - 1], tour[i]
            c, d = tour[j], tour[(j + 1) % n]
            delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
            key = (min(a, c), max(a, c))
            is_tabu = tabu.get(key, -1) > it
            if is_tabu and cur_len + delta >= best_len:   # aspiration criterion
                continue
            if delta < move_delta:
                move, move_delta = (i, j, key), delta
        if move is None:
            continue
        i, j, key = move
        tour[i:j + 1] = reversed(tour[i:j + 1])
        cur_len += move_delta
        tabu[key] = it + tenure
        if cur_len < best_len - 1e-12:
            best, best_len = tour[:], cur_len
        history.append(best_len)
    return best, tour_length(best, D), history


def held_karp(D):
    """Exact optimum length via bitmask DP. Only use for n <= ~13."""
    n = len(D)
    if n < 2:
        return 0.0
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0.0
    for mask in range(1 << n):
        if not mask & 1:
            continue
        for last in range(n):
            cur = dp[mask][last]
            if cur == INF or not (mask >> last) & 1:
                continue
            for nxt in range(n):
                if (mask >> nxt) & 1:
                    continue
                nm = mask | (1 << nxt)
                v = cur + D[last][nxt]
                if v < dp[nm][nxt]:
                    dp[nm][nxt] = v
    full = (1 << n) - 1
    return min(dp[full][last] + D[last][0] for last in range(1, n))
