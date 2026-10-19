"""Replay the season and measure how much time the actual plans gave away.

Usage:
    python scripts/run_backtest.py --lap 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1strategy import backtest, plots  # noqa: E402
from f1strategy.model import load as load_model  # noqa: E402
from f1strategy.predictor import Conditions, DeltaPredictor  # noqa: E402
from f1strategy.runtime import find_race, read_laps, read_pit_losses  # noqa: E402
from f1strategy.strategy import RaceContext  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lap", type=int, default=30)
    args = parser.parse_args()

    laps = read_laps(ROOT / "data" / "processed" / "laps_2024.csv")
    pit_losses = read_pit_losses(ROOT / "data" / "processed" / "pit_losses.csv")
    model = load_model(ROOT / "artifacts" / "lap_delta_model.joblib")

    races = (
        laps[["session_key", "round"]]
        .drop_duplicates("session_key")
        .sort_values("round")
    )
    rows = []
    for _, race_row in races.iterrows():
        race = find_race(laps, pit_losses, str(int(race_row["round"])))
        sample = laps[laps["session_key"] == race.session_key].iloc[0]
        context = RaceContext(
            total_laps=race.total_laps,
            baseline=100.0,
            track_temp=float(sample["track_temp"]),
            air_temp=float(sample["air_temp"]),
            pit_loss=race.pit_loss,
        )
        predictor = DeltaPredictor(model, Conditions(total_laps=race.total_laps))
        rows.extend(backtest.replay_race(laps, race.session_key, predictor, context, lap=args.lap))

    summary = backtest.summarise(rows)
    frame = pd.DataFrame([row.__dict__ for row in rows])
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    frame.to_csv(artifacts / f"backtest_lap{args.lap}.csv", index=False)
    with (artifacts / f"backtest_summary_lap{args.lap}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plots.plot_regret(frame, ROOT / "reports" / "figures" / "strategy_regret.png")
    print(f"Decision lap {args.lap}")
    for key, value in summary.items():
        print(f"  {key:24s} {value:8.3f}")
    print("\nLargest regrets")
    valid = frame[frame["valid_actual"] & frame["clean_after"]].copy()
    valid["regret"] = valid["actual_cost"] - valid["best_cost"]
    columns = ["circuit", "driver", "compound", "tyre_age", "best_plan", "actual_plan", "regret"]
    print(valid.nlargest(6, "regret")[columns].to_string(index=False))


if __name__ == "__main__":
    main()
