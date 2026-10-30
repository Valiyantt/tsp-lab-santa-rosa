"""
santa_rosa.py - Hyper-local delivery route optimiser for Santa Rosa, Laguna.

Uses the from-scratch Genetic Algorithm in ga.py on REAL landmark coordinates
(looked up with Google Places) and changes the problem while it runs:

  gen   0 : baseline map, 10 stops
  gen  60 : traffic congestion on the Santa Rosa-Tagaytay Road corridor
  gen 120 : a new EXPRESS order appears (population is repaired, not restarted)
  gen 180 : flooding makes the City Hall area costly to reach
  gen 240 : traffic clears and the flood subsides

At every phase the GA result is compared with the exact optimum (Held-Karp,
feasible because there are only 10-11 stops) so you can see how close it gets.

  python santa_rosa.py            -> writes output/santa_rosa_* files

ASSUMPTIONS (state these in your report):
  * Distance = straight-line (Haversine) x 1.3 detour factor, NOT real road
    distance. Real road networks need a routing API.
  * Congestion / flood multipliers are scenario parameters, not measured data.
  * The express order location is simulated (a point near Balibago Complex).
"""
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ga import GeneticTSP, build_matrix, tour_length
from baselines import nearest_neighbour, held_karp

OUT = "output"
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "santa_rosa_results.txt"), "w")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


# ----------------------------------------------------------------------------
# Real coordinates (lat, lon) from Google Places
# ----------------------------------------------------------------------------
STOPS = [
    ("Laguna Technopark (Depot)", 14.2613265, 121.0575963),
    ("SM City Sta. Rosa",         14.3115473, 121.0985251),
    ("Robinsons Sta. Rosa",       14.3191667, 121.0966667),
    ("Balibago Complex",          14.2915997, 121.0980124),
    ("Santa Rosa City Hall",      14.3146492, 121.1124535),
    ("Paseo de Santa Rosa",       14.2477814, 121.0631716),
    ("Paseo Outlets (Greenfield)", 14.2485967, 121.0643075),
    ("Vista Mall Sta. Rosa",      14.2434182, 121.0580339),
    ("Ayala Malls Nuvali",        14.2367513, 121.0577278),
    ("QualiMed Hospital",         14.2321549, 121.0528732),
]
EXPRESS = ("EXPRESS order (Balibago, simulated)", 14.2938, 121.1010)

DETOUR = 1.3                      # straight-line -> road distance (assumption)
CORRIDOR = {"Paseo de Santa Rosa", "Paseo Outlets (Greenfield)", "Vista Mall Sta. Rosa",
            "Ayala Malls Nuvali", "QualiMed Hospital"}   # Tagaytay Rd side
CONGESTION = 1.6                  # cost multiplier on links touching the corridor
FLOOD_NODE = "Santa Rosa City Hall"
FLOOD = 3.0                       # cost multiplier on links touching it


def haversine_km(a, b):
    """Great-circle distance in km between (lat, lon) points."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def road_km(a, b):
    return haversine_km(a, b) * DETOUR


class Scenario:
    """Holds the names/points and the currently active conditions."""
    def __init__(self):
        self.names = [s[0] for s in STOPS]
        self.points = [(s[1], s[2]) for s in STOPS]
        self.traffic = False
        self.flood = False

    def cost_matrix(self):
        D = build_matrix(self.points, road_km)
        n = len(D)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                f = 1.0
                if self.traffic and (self.names[i] in CORRIDOR or self.names[j] in CORRIDOR):
                    f *= CONGESTION
                if self.flood and FLOOD_NODE in (self.names[i], self.names[j]):
                    f *= FLOOD
                D[i][j] *= f
        return D


def main():
    sc = Scenario()
    ga = GeneticTSP(sc.points, pop_size=80, seed=7, dist_fn=road_km, local_rate=0.05)
    ga.set_distances(sc.cost_matrix())

    events = {
        60:  ("TRAFFIC: Tagaytay-Sta. Rosa Rd congested (x%.1f cost)" % CONGESTION, "traffic_on"),
        120: ("NEW ORDER: express delivery near Balibago", "add"),
        180: ("FLOOD: City Hall area costly (x%.1f cost)" % FLOOD, "flood_on"),
        240: ("CLEARED: traffic and flood back to normal", "clear"),
    }
    total = 300
    hist, frames, markers, summary = [], [], [], []

    def snapshot(label):
        D = ga.D
        _, L = ga.best()
        nn = tour_length(nearest_neighbour(D), D)
        opt = held_karp(D)
        summary.append((label, len(ga.points), L, opt, nn))
        log(f"  {label:<44} stops={len(ga.points):2d}  GA={L:7.2f}  optimal={opt:7.2f}  "
            f"gap={max(0.0, 100 * (L - opt) / opt):5.2f}%  nearest-neighbour={nn:7.2f}")

    log("=" * 100)
    log("SANTA ROSA, LAGUNA - dynamic delivery routing with a Genetic Algorithm")
    log("Costs are effective km (straight-line x 1.3 detour x scenario multipliers)")
    log("=" * 100)
    t0 = time.time()
    for g in range(total):
        if g in events:
            text, kind = events[g]
            snapshot("before: " + ("start" if g == 0 else text.split(":")[0].lower()))
            if kind == "traffic_on":
                sc.traffic = True
            elif kind == "flood_on":
                sc.flood = True
            elif kind == "clear":
                sc.traffic = sc.flood = False
            elif kind == "add":
                sc.names.append(EXPRESS[0])
                sc.points.append((EXPRESS[1], EXPRESS[2]))
                ga.add_city(sc.points[-1])
            ga.set_distances(sc.cost_matrix())
            markers.append((g, text))
            log(f"gen {g:3d}: {text}")
        ga.step()
        _, L = ga.best()
        hist.append(L)
        if g % 4 == 0:
            frames.append((g, list(sc.points), list(sc.names), ga.best()[0], L,
                           sc.traffic, sc.flood,
                           next((m[1] for m in markers if 0 <= g - m[0] < 24), "")))
    snapshot("end of run (all clear)")
    log(f"\nTotal evolution time: {time.time() - t0:.1f} s for {total} generations")

    # best final route
    tour, L = ga.best()
    start = tour.index(0)
    tour = tour[start:] + tour[:start]
    log("\nFinal route (depot first):")
    for i in tour:
        log(f"  -> {sc.names[i]}")
    log(f"  -> {sc.names[0]} (return)   total = {L:.2f} effective km")

    # ---------------- plots ----------------
    def draw_map(ax, pts, names, tr, length, traffic=False, flood=False, title=""):
        lat0 = math.radians(14.28)
        ax.set_aspect(1 / math.cos(lat0))
        if traffic:
            cx = [p[1] for p, n in zip(pts, names) if n in CORRIDOR]
            cy = [p[0] for p, n in zip(pts, names) if n in CORRIDOR]
            ax.scatter(cx, cy, s=900, c="orange", alpha=.25, zorder=1)
        xs = [pts[c][1] for c in tr] + [pts[tr[0]][1]]
        ys = [pts[c][0] for c in tr] + [pts[tr[0]][0]]
        ax.plot(xs, ys, "-", color="tab:blue", lw=1.8, zorder=2)
        for k, (p, n) in enumerate(zip(pts, names)):
            col = "tab:green" if k == 0 else ("magenta" if n.startswith("EXPRESS") else "tab:red")
            if flood and n == FLOOD_NODE:
                col = "navy"
            ax.scatter(p[1], p[0], c=col, s=70 if k else 140, marker="*" if k == 0 else "o", zorder=3)
            ax.annotate(n.replace(" (Depot)", ""), (p[1], p[0]), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.3)

    fig, ax = plt.subplots(figsize=(7.5, 8))
    draw_map(ax, sc.points, sc.names, tour, L, title=f"Optimised delivery route, Santa Rosa, Laguna ({L:.1f} effective km)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "santa_rosa_route.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist, color="tab:blue")
    for g, text in markers:
        ax.axvline(g, color="gray", ls=":")
        ax.text(g + 2, max(hist) * 0.995, text.split(":")[0], rotation=90, va="top", fontsize=8)
    ax.set(title="GA best route cost over time (cost jumps are real: the world changed)",
           xlabel="Generation", ylabel="Effective km")
    ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "santa_rosa_timeline.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(7, 7.5))

    def draw(i):
        ax.clear()
        g, pts, names, tr, length, traffic, flood, note = frames[i]
        draw_map(ax, pts, names, tr, length, traffic, flood,
                 title=f"Gen {g} | stops {len(pts)} | cost {length:.1f} km" + (f"\n{note}" if note else ""))

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=150)
    anim.save(os.path.join(OUT, "santa_rosa_simulation.gif"), writer=PillowWriter(fps=7))
    plt.close()
    log("\nSaved: santa_rosa_route.png, santa_rosa_timeline.png, santa_rosa_simulation.gif")


if __name__ == "__main__":
    main()
