"""Example_behaviour -- assemble the figure (thin wrapper around build.py).

Run on haas after compute.py (repo root):
    ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Example_behaviour/make_figure.py [--date YYYY-MM-DD]
"""

import argparse
import sys
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import build  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    build.build(FIG_DIR, parser.parse_args().date)
