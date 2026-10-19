"""Feature preparation for the tyre degradation model.

Absolute lap times are a bad target. They carry the circuit layout, the car's
pace, the fuel load and the amount of rubber on the track, none of which a
model can transfer to a race it has never seen.

This module isolates the part a strategy decision actually depends on: how much
a stint slows down as the tyres wear. Each lap is first corrected for fuel, then
expressed relative to the pace of the same stint on fresh tyres. Everything
that is common to all strategies cancels out of that comparison, which is why
the simulator can rank pit windows using only the degradation curve.

The fuel sensitivity is estimated from the data in
:func:`estimate_fuel_effect`. Fresh tyre laps give 0.0359 s per kg on the 2024
sample, or about 3.9 s over a full 110 kg load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
FUEL_CAPACITY_KG = 110.0
FUEL_EFFECT_S_PER_KG = 0.0359
FEATURES = [
    "tyre_age",
    "tyre_age_sq",
    "track_temp_delta",
    "air_temp_delta",
    "compound_hard",
    "compound_medium",
    "compound_soft",
]
TARGET = "stint_delta"


def fuel_mass(laps: pd.DataFrame) -> pd.Series:
    """Approximate fuel on board at the start of a lap, in kilograms."""
    remaining = laps["total_laps"] - laps["lap_number"] + 1
    return FUEL_CAPACITY_KG * remaining.clip(lower=0) / laps["total_laps"].clip(lower=1)


def estimate_fuel_effect(clean: pd.DataFrame, max_tyre_age: int = 4) -> float:
    """Estimate seconds lost per kilogram of fuel from fresh tyre laps.

    The regression removes session and compound means first, so only the
    within race variation in fuel load is used. Track evolution remains a
    confounder, but repeating the estimate at different tyre ages gives a
    stable answer, which suggests degradation is not driving it.
    """
    fresh = clean[clean["tyre_age"] <= max_tyre_age].copy()
    fresh["fuel"] = fuel_mass(fresh)
    grouped = fresh.groupby(["session_key", "compound"])
    time_demeaned = fresh["lap_duration"] - grouped["lap_duration"].transform("mean")
    fuel_demeaned = fresh["fuel"] - grouped["fuel"].transform("mean")
    denominator = float((fuel_demeaned**2).sum())
    if denominator == 0:
        return FUEL_EFFECT_S_PER_KG
    return float((time_demeaned * fuel_demeaned).sum() / denominator)


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Drop laps that do not represent dry, flat out running."""
    frame = laps.copy()
    # A damp track has a different grip level that a dry tyre model cannot
    # describe, so whole sessions with meaningful rain are left out.
    rain_share = frame.groupby("session_key")["rainfall"].transform(
        lambda values: values.fillna(0).gt(0).mean()
    )
    mask = (
        rain_share.le(0.10)
        & ~frame["is_pit_in_lap"]
        & ~frame["is_pit_out_lap"]
        & ~frame["is_safety_car_lap"]
        & ~frame["is_slow_lap"]
        & frame["lap_number"].gt(1)
        & frame["compound"].isin(COMPOUNDS)
        & frame["tyre_age"].notna()
        & frame["lap_duration"].gt(0)
        & frame["rainfall"].fillna(0).eq(0)
        & frame["total_laps"].gt(0)
    )
    cleaned = frame.loc[mask].copy()
    cleaned["tyre_age"] = cleaned["tyre_age"].astype(int)
    return cleaned.reset_index(drop=True)


def add_stint_delta(
    clean: pd.DataFrame,
    fuel_effect: float = FUEL_EFFECT_S_PER_KG,
) -> pd.DataFrame:
    """Attach fuel corrected laps and the within stint degradation delta."""
    frame = clean.copy()
    frame["fuel_mass"] = fuel_mass(frame)
    frame["corrected_lap"] = frame["lap_duration"] - fuel_effect * frame["fuel_mass"]

    key = ["session_key", "driver_number", "stint_number"]
    fresh = frame[frame["tyre_age"] <= 2]
    reference = (
        fresh.groupby(key)["corrected_lap"].median().rename("stint_reference").reset_index()
    )
    counts = frame.groupby(key)["corrected_lap"].size().rename("stint_laps").reset_index()
    frame = frame.merge(reference, on=key, how="left").merge(counts, on=key, how="left")
    frame = frame[frame["stint_reference"].notna() & frame["stint_laps"].ge(3)]
    frame[TARGET] = frame["corrected_lap"] - frame["stint_reference"]

    baseline = (
        frame.groupby(["session_key", "driver_number"])["corrected_lap"]
        .quantile(0.02)
        .rename("baseline")
        .reset_index()
    )
    frame = frame.merge(baseline, on=["session_key", "driver_number"], how="left")
    frame = frame[frame[TARGET].between(-2, 8)]
    return frame.reset_index(drop=True)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the model features. Works on a frame with the stint delta."""
    out = frame.copy()
    out["tyre_age_sq"] = out["tyre_age"] ** 2
    out["laps_remaining"] = out["total_laps"] - out["lap_number"]
    out["lap_fraction"] = out["lap_number"] / out["total_laps"].clip(lower=1)
    for compound in COMPOUNDS:
        out[f"compound_{compound.lower()}"] = out["compound"].eq(compound).astype(int)
    for column in ["track_temp", "air_temp"]:
        median = out.groupby("session_key")[column].transform("median")
        out[f"{column}_delta"] = out[column].fillna(median).sub(median).fillna(0.0)
    out["tyre_age"] = out["tyre_age"].astype(int)
    return out


def prepare(laps: pd.DataFrame) -> pd.DataFrame:
    """Clean, correct, baseline and featurise a raw lap table in one call."""
    return add_features(add_stint_delta(clean_laps(laps)))


def feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Return a model ready matrix and keep the column order stable."""
    return rows.loc[:, FEATURES].astype(float)


def summarise(frame: pd.DataFrame) -> dict[str, float]:
    """Describe the modelling sample for the report."""
    return {
        "laps": float(len(frame)),
        "races": float(frame["session_key"].nunique()),
        "drivers": float(frame["driver_number"].nunique()),
        "stints": float(
            frame[["session_key", "driver_number", "stint_number"]].drop_duplicates().shape[0]
        ),
        "median_delta": float(np.median(frame[TARGET])),
        "p95_delta": float(np.percentile(frame[TARGET], 95)),
    }
