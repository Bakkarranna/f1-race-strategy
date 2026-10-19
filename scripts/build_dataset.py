"""Download the 2024 season and write the processed lap table.

Usage:
    python scripts/build_dataset.py --year 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1strategy.dataset import build_season  # noqa: E402
from f1strategy.openf1 import OpenF1Client  # noqa: E402
from f1strategy.pitloss import circuit_losses  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    parser.add_argument("--cache", default=".cache/openf1")
    parser.add_argument("--output", default="data/processed/laps_2024.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = OpenF1Client(cache_dir=args.cache)
    laps = build_season(client, args.year, rounds=args.rounds)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    laps.to_csv(output, index=False)

    losses = circuit_losses(laps)
    loss_path = output.parent / "pit_losses.csv"
    losses.to_csv(loss_path, index=False)

    print(f"Laps: {len(laps)}")
    print(f"Races: {laps['session_key'].nunique()}")
    print(f"Drivers: {laps['driver_number'].nunique()}")
    print(f"Written: {output}")
    print(losses.to_string(index=False))
    print(f"Written: {loss_path}")


if __name__ == "__main__":
    main()
