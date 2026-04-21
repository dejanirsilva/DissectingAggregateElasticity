#!/usr/bin/env python3
"""
Download US nonfinancial corporate leverage series and compute post-1980
averages.

Series:
  - NCBCMDPMVCE (FL104104016.Q): Debt as % of market value of corporate equities
  - NCBCMDPNWHC (FL104104026.Q): Debt as % of net worth (historical cost)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"
SUMMARY_CSV_PATH = PROCESSED_DIR / "average_since_1980.csv"
SERIES = {
    "NCBCMDPMVCE": {
        "flow_of_funds_code": "FL104104016.Q",
        "label": "Debt as % of market value of corporate equities",
    },
    "NCBCMDPNWHC": {
        "flow_of_funds_code": "FL104104026.Q",
        "label": "Debt as % of net worth (historical cost)",
    },
}


@dataclass(frozen=True)
class Observation:
    observation_date: date
    value: float


def download_csv(series_id: str) -> str:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def parse_observations(series_id: str, csv_text: str) -> list[Observation]:
    rows = list(csv.DictReader(csv_text.splitlines()))
    out: list[Observation] = []
    for row in rows:
        value_raw = (row.get(series_id) or "").strip()
        if not value_raw or value_raw == ".":
            continue
        out.append(
            Observation(
                observation_date=datetime.strptime(
                    row["observation_date"], "%Y-%m-%d"
                ).date(),
                value=float(value_raw),
            )
        )
    return out


def write_raw_csv(series_id: str, observations: list[Observation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["observation_date", series_id])
        for obs in observations:
            writer.writerow([obs.observation_date.isoformat(), f"{obs.value:.12f}"])


def compute_average_since(observations: list[Observation], start: date) -> tuple[float, int]:
    sample = [obs.value for obs in observations if obs.observation_date >= start]
    if not sample:
        raise ValueError(f"No observations found on or after {start.isoformat()}.")
    return statistics.fmean(sample), len(sample)


def write_summary_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "series_id",
                "flow_of_funds_code",
                "series_label",
                "start_date",
                "end_date",
                "n_obs",
                "average_percent",
                "average_ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["series_id"],
                    row["flow_of_funds_code"],
                    row["series_label"],
                    row["start_date"],
                    row["end_date"],
                    row["n_obs"],
                    row["average_percent"],
                    row["average_ratio"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download debt-to-equity ratio and compute average since a start year."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1980,
        help="Start year for average calculation (default: 1980).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = date(args.start_year, 1, 1)
    summary_rows: list[dict[str, str]] = []

    for series_id, meta in SERIES.items():
        csv_text = download_csv(series_id)
        observations = parse_observations(series_id, csv_text)
        if not observations:
            raise ValueError(f"No observations were parsed for {series_id}.")

        raw_csv_path = RAW_DIR / f"{series_id}.csv"
        write_raw_csv(series_id, observations, raw_csv_path)

        avg, n_obs = compute_average_since(observations, start_date)
        summary_rows.append(
            {
                "series_id": series_id,
                "flow_of_funds_code": meta["flow_of_funds_code"],
                "series_label": meta["label"],
                "start_date": max(start_date, observations[0].observation_date).isoformat(),
                "end_date": observations[-1].observation_date.isoformat(),
                "n_obs": str(n_obs),
                "average_percent": f"{avg:.6f}",
                "average_ratio": f"{avg / 100.0:.6f}",
            }
        )
        print(f"Downloaded {len(observations)} observations to: {raw_csv_path}")
        print(
            f"Average {series_id} since {start_date.isoformat()}: "
            f"{avg:.4f}% ({avg / 100.0:.4f} ratio, n={n_obs})"
        )

    write_summary_csv(summary_rows, SUMMARY_CSV_PATH)
    print(f"Wrote summary: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
