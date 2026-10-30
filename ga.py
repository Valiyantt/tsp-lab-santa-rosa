"""
ga.py - Genetic Algorithm for the Traveling Salesman Problem (from scratch).

Representation : a tour is a permutation of city indices [0..n-1].
Selection      : tournament selection (size k).
Crossover      : Order Crossover (OX) - preserves relative order, always valid.
Local search   : bounded 2-opt on a fraction of children (memetic GA).
Mutation       : inversion (reverses a segment) or swap, chosen at random.
Elitism        : the best `elite` tours survive unchanged every generation.

Dynamic support: cities can be added/removed WHILE the population evolves.
Instead of restarting, every existing tour is repaired (cheapest insertion for
a new city, deletion + relabel for a removed one), so earlier progress is kept.

Only the Python standard library is used (no library calls for core logic).
"""
import math
import random


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_matrix(points, dist_fn=euclid):
    """Full symmetric distance matrix, O(n^2) time and space."""
    n = len(points)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(points[i], points[j])
            D[i][j] = D[j][i] = d
    return D


def tour_length(tour, D):
    """Length of the closed tour, O(n)."""
    total = 0.0
    prev = tour[-1]
    for c in tour:
        total += D[prev][c]
        prev = c
    return total


def order_crossover(p1, p2, rng):
    """OX: copy a slice from p1, fill the rest in the order they appear in p2."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2)) if n > 1 else (0, 0)
    child = [-1] * n
    child[a:b + 1] = p1[a:b + 1]
    used = set(child[a:b + 1])
    pos = (b + 1) % n
    for gene in p2[b + 1:] + p2[:b + 1]:
        if gene not in used:
            child[pos] = gene
            pos = (pos + 1) % n
    return child


def mutate(tour, rng):
    n = len(tour)
    if n < 3:
        return
    i, j = sorted(rng.sample(range(n), 2))
    if rng.random() < 0.7:          # inversion (a 2-opt style move)
        tour[i:j + 1] = reversed(tour[i:j + 1])
    else:                            # swap
        tour[i], tour[j] = tour[j], tour[i]


class GeneticTSP:
    def __init__(self, points, pop_size=150, elite=2, tournament_k=5,
                 crossover_rate=0.9, mutation_rate=0.25, seed=None,
                 greedy_seed=True, local_rate=0.15, dist_fn=euclid):
        self.rng = random.Random(seed)
        self.points = list(points)
        self.dist_fn = dist_fn          # e.g. Haversine for GPS points
        self.D = build_matrix(self.points, dist_fn)
        self.pop_size = pop_size
        self.elite = elite
        self.k = tournament_k
        self.cx = crossover_rate
        self.mut = mutation_rate
        self.local_rate = local_rate   # fraction of children refined by 2-opt
        self.generation = 0
        self.evaluations = 0
        n = len(self.points)
        self.pop = []
        for _ in range(pop_size):
            t = list(range(n))
            self.rng.shuffle(t)
            self.pop.append(t)
        if greedy_seed and n > 2:           # one nearest-neighbour tour seeds quality
            self.pop[0] = self._nearest_neighbour(self.rng.randrange(n))
        self._score()

    # ---- helpers -------------------------------------------------------
    def _nearest_neighbour(self, start):
        n = len(self.points)
        left = set(range(n))
        left.remove(start)
        tour = [start]
        while left:
            last = tour[-1]
            nxt = min(left, key=lambda c: self.D[last][c])
            left.remove(nxt)
            tour.append(nxt)
        return tour

    def _score(self):
        self.fit = [tour_length(t, self.D) for t in self.pop]
        self.evaluations += len(self.pop)

    def _tournament(self):
        best = None
        for _ in range(self.k):
            i = self.rng.randrange(len(self.pop))
            if best is None or self.fit[i] < self.fit[best]:
                best = i
        return self.pop[best]

    def _two_opt(self, t, max_passes=3):
        """Bounded first-improvement 2-opt (memetic step), O(n^2) per pass."""
        D, n = self.D, len(t)
        for _ in range(max_passes):
            improved = False
            for i in range(n - 1):
                a, b = t[i], t[i + 1]
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    c, d = t[j], t[(j + 1) % n]
                    if D[a][c] + D[b][d] < D[a][b] + D[c][d] - 1e-12:
                        t[i + 1:j + 1] = reversed(t[i + 1:j + 1])
                        a, b = t[i], t[i + 1]
                        improved = True
            if not improved:
                break
        return t

    # ---- evolution -----------------------------------------------------
    def step(self):
        """Advance one generation."""
        order = sorted(range(len(self.pop)), key=self.fit.__getitem__)
        new_pop = [self.pop[i][:] for i in order[:self.elite]]
        while len(new_pop) < self.pop_size:
            p1, p2 = self._tournament(), self._tournament()
            child = (order_crossover(p1, p2, self.rng)
                     if self.rng.random() < self.cx else p1[:])
            if self.rng.random() < self.mut:
                mutate(child, self.rng)
            if self.rng.random() < self.local_rate and len(child) > 3:
                self._two_opt(child)
            new_pop.append(child)
        self.pop = new_pop
        self._score()
        self.generation += 1

    def run(self, generations):
        history = []
        for _ in range(generations):
            self.step()
            history.append(self.best()[1])
        return history

    def best(self):
        i = min(range(len(self.pop)), key=self.fit.__getitem__)
        return self.pop[i][:], self.fit[i]

    # ---- dynamic changes ----------------------------------------------
    def add_city(self, point):
        """Add a city; insert it at its cheapest position in every tour."""
        new = len(self.points)
        self.points.append(point)
        for row in self.D:
            row.append(0.0)
        self.D.append([0.0] * (new + 1))
        for i in range(new):
            d = self.dist_fn(self.points[i], point)
            self.D[i][new] = self.D[new][i] = d
        D = self.D
        for t in self.pop:
            best_pos, best_cost = 0, float("inf")
            for p in range(len(t)):
                a, b = t[p - 1], t[p]
                cost = D[a][new] + D[new][b] - D[a][b]
                if cost < best_cost:
                    best_cost, best_pos = cost, p
            t.insert(best_pos, new)
        self._score()
        return new

    def remove_city(self, idx):
        """Remove city idx; delete from every tour and relabel higher indices."""
        del self.points[idx]
        del self.D[idx]
        for row in self.D:
            del row[idx]
        for t in self.pop:
            t.remove(idx)
            for p, c in enumerate(t):
                if c > idx:
                    t[p] = c - 1
        self._score()

    def set_distances(self, D):
        """Replace the whole cost matrix (traffic, closures...) and re-score.
        Tours stay valid; only their lengths change, so the population keeps
        its structure and simply re-adapts to the new costs."""
        self.D = D
        self._score()
