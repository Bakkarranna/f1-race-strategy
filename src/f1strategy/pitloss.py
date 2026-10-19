"""Pit stop loss estimation.

A pit stop costs the time spent in the pit lane plus the time lost slowing down
and getting back up to speed. The cleanest estimate from lap data is the sum of
the pit in lap and the pit out lap minus two representative green laps for that
driver. The pit lane transit time reported by the timing system is kept as a
sanity check: total loss should always exceed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import COMPOUNDS


def stop_losses(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per pit stop with the estimated time lost."""
    frame = raw.copy()
    frame["key"] = frame["session_key"].astype(str) + "-" + frame["driver_number"].astype(str)

    clean = frame[
        ~frame["is_pit_in_lap"]
        & ~frame["is_pit_out_lap"]
        & ~frame["is_slow_lap"]
        & ~frame["is_safety_car_lap"]
        & frame["lap_number"].gt(1)
        & frame["compound"].isin(COMPOUNDS)
    ]
    session_reference = clean.groupby("key")["lap_duration"].median().rename("session_reference")
    frame = frame.merge(session_reference, on="key", how="left")

    rows = []
    pit_rows = frame[frame["is_pit_in_lap"] & frame["pit_lane_duration"].notna()]
    for _, row in pit_rows.iterrows():
        out_lap = frame[
            (frame["session_key"] == row["session_key"])
            & (frame["driver_number"] == row["driver_number"])
            & (frame["lap_number"] == row["lap_number"] + 1)
        ]
        if out_lap.empty or pd.isna(row["session_reference"]):
            continue

        # Local reference laps avoid mixing the fuel effect of an early stop
        # with a mid race median. They also keep safety car laps out.
        nearby = clean[
            (clean["session_key"] == row["session_key"])
            & (clean["driver_number"] == row["driver_number"])
            & (clean["lap_number"].sub(row["lap_number"]).abs().le(4))
        ]
        reference = (
            float(nearby["lap_duration"].median())
            if len(nearby) >= 2
            else float(row["session_reference"])
        )
        out_time = float(out_lap["lap_duration"].iloc[0])
        loss = float(row["lap_duration"]) + out_time - 2 * reference
        if not 10 <= loss <= 45:
            continue
        rows.append(
            {
                "session_key": int(row["session_key"]),
                "round": int(row["round"]),
                "circuit": row["circuit"],
                "driver_number": int(row["driver_number"]),
                "lap_number": int(row["lap_number"]),
                "pit_lane_duration": float(row["pit_lane_duration"]),
                "loss": loss,
                "reference_lap": reference,
            }
        )
    return pd.DataFrame(rows)


def circuit_losses(raw: pd.DataFrame) -> pd.DataFrame:
    """Median pit loss per circuit, with the spread across stops."""
    stops = stop_losses(raw)
    if stops.empty:
        return pd.DataFrame(columns=["circuit", "pit_loss", "stops", "pit_lane"])
    grouped = (
        stops.groupby("circuit")
        .agg(
            pit_loss=("loss", "median"),
            pit_loss_iqr=("loss", lambda values: np.percentile(values, 75) - np.percentile(values, 25)),
            stops=("loss", "size"),
            pit_lane=("pit_lane_duration", "median"),
        )
        .reset_index()
        .sort_values("pit_loss")
    )
    return grouped
