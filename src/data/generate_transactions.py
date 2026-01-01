"""Generate synthetic UPI-like transaction dataset (2020-2025).

This script creates a deterministic synthetic dataset of digital payment transactions
with realistic features and a deterministic failure rule that depends on:
- peak hours
- network latency
- bank load score
- retry count
- bank reliability differences

The generated file is saved to: data/synthetic/upi_transactions_2020_2025.csv

Usage:
    python generate_transactions.py --n 200000 --start-year 2020 --end-year 2025

Notes:
- The failure flag is a deterministic function of transaction features
  (no random coin flips). The random seed fixes generated features so results
  are reproducible.
- No ML training or visualization is performed here.
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import timedelta

import numpy as np
import pandas as pd


RNG_SEED = 42  # reproducible synthetic data


def _make_bank_profiles() -> dict:
    """Return Indian banks with base failure and base load scores.

    Lower `base_fail_rate` means more reliable bank. `base_load` is the
    typical normalized load (0-1) that affects failures and latency.

    Replaced generic banks with Indian banks while preserving reliability
    differences (lower base_fail_rate => more reliable).
    """
    banks = {
        "SBI": {"prob": 0.28, "base_fail_rate": 0.004, "base_load": 0.33},
        "HDFC": {"prob": 0.22, "base_fail_rate": 0.008, "base_load": 0.38},
        "ICICI": {"prob": 0.18, "base_fail_rate": 0.015, "base_load": 0.50},
        "Axis": {"prob": 0.12, "base_fail_rate": 0.030, "base_load": 0.60},
        "Kotak": {"prob": 0.10, "base_fail_rate": 0.060, "base_load": 0.70},
        "PNB": {"prob": 0.06, "base_fail_rate": 0.095, "base_load": 0.80},
        "YesBank": {"prob": 0.04, "base_fail_rate": 0.12, "base_load": 0.85},
    }
    return banks


def _hour_weights() -> np.ndarray:
    """Return normalized hourly weights reflecting peak and off-peak hours.

    Peak hours (morning commute and evening) have higher weights.
    """
    base = np.ones(24, dtype=float)
    # Define peak hours (local usage peaks): morning and evening
    morning_peak = list(range(8, 12))  # 08:00-11:59
    evening_peak = list(range(18, 22))  # 18:00-21:59
    for h in morning_peak + evening_peak:
        base[h] *= 3.5
    # Slight bump in lunch hour
    base[12] *= 1.3
    return base / base.sum()


def _dow_weights() -> np.ndarray:
    """Return normalized day-of-week weights to slightly reduce weekend volume."""
    # Monday=0 ... Sunday=6
    weights = np.array([1.05, 1.05, 1.05, 1.02, 1.02, 0.85, 0.75], dtype=float)
    return weights / weights.sum()


def generate_transactions(
    n: int = 200_000,
    start_year: int = 2020,
    end_year: int = 2025,
    out_path: str | None = None,
) -> pd.DataFrame:
    """Generate synthetic transactions and return as a DataFrame.

    The function is vectorized and uses a fixed RNG seed for reproducibility.

    Failure is determined by a deterministic score computed from features.
    """
    rng = np.random.default_rng(RNG_SEED)

    # Prepare date candidates and sampling weights
    date_index = pd.date_range(
        start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D"
    )
    # Day-of-week weights
    dow_w = _dow_weights()
    date_weights = np.array([dow_w[dt.dayofweek] for dt in date_index], dtype=float)
    date_weights = date_weights / date_weights.sum()

    # Hour weights
    hour_w = _hour_weights()

    # Sample dates and hours for each transaction
    chosen_days = rng.choice(date_index, size=n, p=date_weights)
    chosen_hours = rng.choice(np.arange(24), size=n, p=hour_w)
    # sample minute/second uniformly
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)

    timestamps = pd.to_datetime(chosen_days) + pd.to_timedelta(chosen_hours, unit="h")
    timestamps += pd.to_timedelta(minutes, unit="m") + pd.to_timedelta(
        seconds, unit="s"
    )

    # Amounts: many small transactions, fewer large ones (log-normal)
    amount = np.clip(rng.lognormal(mean=3.0, sigma=1.2, size=n), 1.0, 25000.0)
    # Round to cents
    amount = np.round(amount, 2)

    # Bank assignment and bank-related features
    banks = _make_bank_profiles()
    bank_names = list(banks.keys())
    bank_probs = np.array([banks[b]["prob"] for b in bank_names], dtype=float)
    bank_probs = bank_probs / bank_probs.sum()
    chosen_banks = rng.choice(bank_names, size=n, p=bank_probs)

    # Bank base fail rate and statically expected load
    bank_base_fail = np.array([banks[b]["base_fail_rate"] for b in chosen_banks])
    bank_base_load = np.array([banks[b]["base_load"] for b in chosen_banks])

    # Bank load score fluctuates around base (clipped 0-1)
    bank_load_score = np.clip(
        rng.normal(loc=bank_base_load, scale=0.12, size=n), 0.0, 1.0
    )

    # Device types and device base latencies
    device_types = np.array(["mobile", "desktop", "pos"])
    device_probs = np.array([0.82, 0.12, 0.06])
    chosen_device = rng.choice(device_types, size=n, p=device_probs)
    device_base_latency = np.select(
        [chosen_device == "mobile", chosen_device == "desktop", chosen_device == "pos"],
        [120.0, 60.0, 40.0],
    )

    # Network latency (ms): base + bank_load component + random exponential tail
    network_latency_ms = (
        device_base_latency
        + bank_load_score * 900.0
        + rng.exponential(scale=50.0, size=n)
    )
    network_latency_ms = np.round(network_latency_ms).astype(int)

    # Retry count distribution (more 0s, fewer higher retries)
    retry_probs = np.array([0.86, 0.10, 0.03, 0.01])
    retry_count = rng.choice([0, 1, 2, 3], size=n, p=retry_probs)

    # Past user failure rate for the payer (0-1), beta skewed towards low values
    past_user_failure_rate = rng.beta(a=1.0, b=30.0, size=n)
    past_user_failure_rate = np.round(past_user_failure_rate, 4)

    # Compute derived temporal fields
    year = timestamps.year
    month = timestamps.month
    hour = timestamps.hour
    day_of_week = timestamps.dayofweek  # Monday=0

    # Now compute a deterministic failure score using weighted components.
    # Components:
    #  - peak hour indicator (morning/evening)
    #  - normalized latency (0-1)
    #  - bank load score (0-1)
    #  - retry count normalized (0-1)
    #  - bank intrinsic reliability (higher base_fail increases score)
    #  - past user failure rate (0-1)
    peak_hours = set(list(range(8, 12)) + list(range(18, 22)))
    peak_flag = np.isin(hour, np.array(list(peak_hours), dtype=int)).astype(float)

    norm_latency = np.clip(network_latency_ms / 2000.0, 0.0, 1.0)
    retry_norm = np.clip(retry_count / 5.0, 0.0, 1.0)

    # bank_base_fail is small (e.g., 0.005). Scale it up to have visible effect.
    bank_rel_score = np.clip(bank_base_fail * 6.0, 0.0, 1.0)

    # Weighted sum -> higher means more likely to fail. Deterministic threshold used.
    failure_score = (
        peak_flag * 0.22
        + norm_latency * 0.32
        + bank_load_score * 0.18
        + retry_norm * 0.14
        + bank_rel_score * 0.30
        + past_user_failure_rate * 0.08
    )

    # Cap the score to avoid extreme probabilities and apply a global scaling
    # factor so we can reduce overall failure probability to realistic levels.
    # We keep the original `failure_score` composition intact (no change to
    # features or their weights) and only transform the score into a
    # probability for sampling.
    failure_score = np.minimum(failure_score, 0.95)

    # Choose a target overall failure rate in the realistic range (4%-7%).
    # Compute a global scale that reduces the mean of `failure_score` toward
    # the target. We only scale downward (never increase) to avoid creating
    # unrealistically high failure rates.
    target_rate = 0.055  # target 5.5% overall failures (midpoint of 4%-7%)
    mean_score = float(np.mean(failure_score))
    if mean_score > 0:
        global_scale = min(1.0, target_rate / mean_score)
    else:
        global_scale = 1.0

    # Apply the global scale and ensure the result is a valid probability (0-0.95)
    scaled_score = np.clip(failure_score * global_scale, 0.0, 0.95)

    # Perform a probabilistic sampling (binomial draw) using the scaled score
    # as the success probability for each transaction. Using the same RNG
    # guarantees reproducible draws.
    failure_flag = rng.binomial(1, scaled_score).astype(int)

    # Build the DataFrame
    df = pd.DataFrame(
        {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "timestamp": timestamps,
            "year": year,
            "month": month,
            "hour": hour,
            "day_of_week": day_of_week,
            "amount": amount,
            "bank_name": chosen_banks,
            "network_latency_ms": network_latency_ms,
            "device_type": chosen_device,
            "retry_count": retry_count,
            "bank_load_score": np.round(bank_load_score, 3),
            "past_user_failure_rate": past_user_failure_rate,
            "failure_flag": failure_flag,
        }
    )

    # Save to CSV if requested
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)

    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic UPI-like transactions CSV"
    )
    p.add_argument(
        "--n", type=int, default=200_000, help="number of transactions to generate"
    )
    p.add_argument(
        "--start-year", type=int, default=2020, help="start year (inclusive)"
    )
    p.add_argument("--end-year", type=int, default=2025, help="end year (inclusive)")
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "synthetic", "upi_transactions_2020_2025.csv"),
        help="output CSV path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_transactions(
        n=args.n,
        start_year=args.start_year,
        end_year=args.end_year,
        out_path=args.output,
    )

    # Print a short summary
    total = len(df)
    failures = int(df["failure_flag"].sum())
    print(
        f"Generated {total:,} transactions. Failures: {failures} ({failures/total:.2%})"
    )
    print(f"Saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
