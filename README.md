# TSP Lab: evolve the shortest route

An interactive Genetic Algorithm for the dynamic Traveling Salesman Problem, with a real Santa Rosa, Laguna
delivery scenario (traffic, flooding, express orders).

**Live demo:** https://YOUR-USERNAME.github.io/tsp-lab-santa-rosa/   (replace YOUR-USERNAME)

## What is here
| Path | What it is |
|---|---|
| `index.html` | The whole interactive app (one self-contained file, no build step) |
| `project/` | Python code, tests, results, and the Google Colab notebook |
| `project/GA_TSP_Colab_Notebook.ipynb` | Step-by-step notebook that recreates every result |

## Run the Python code
    cd project
    python -m unittest discover -s tests -v
    python simulate.py
    python scenarios.py
    python algo_tests.py

Pure Python for the algorithms; plots need `matplotlib` and `pillow`.

## Notes
Santa Rosa distances are straight-line x 1.3 (an assumption, not road distance); traffic and flood multipliers
are scenario parameters. Time-limited results vary a little from run to run.
