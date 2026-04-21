#!/usr/bin/env python3
"""
Replicate Figure 6 (Quarterly flows and return volatility).

Inputs:
  1) Processed quarterly flow series from FoF pipeline (raw_data.csv)
  2) CRSP-based quarterly realized volatility series (crsp_quarterly_realized_vol.csv)

Notes:
  - Volatility is computed from CRSP daily value-weighted returns in
    compute_crsp_quarterly_vol.py and annualized there.
  - The FoF "flow" variable is already scaled by lagged market capitalization in
    the existing project pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "raw_data.csv"
CRSP_VOL_PATH = ROOT / "crsp_quarterly_realized_vol.csv"
OUT_DATA_PATH = ROOT / "figure6_rebuilt_data.csv"
OUT_FIG_PATH = ROOT / "figs" / "figure6_from_scratch.png"


def quarter_date_from_yyyymm(yyyymm: pd.Series) -> pd.Series:
    yyyymm_str = yyyymm.astype(int).astype(str)
    year = yyyymm_str.str[:4].astype(int)
    month = yyyymm_str.str[4:].astype(int)
    return pd.to_datetime({"year": year, "month": month, "day": 1})


def prep_data() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH).copy()
    crsp = pd.read_csv(CRSP_VOL_PATH).copy()

    raw = raw.assign(yyyymm=raw["yyyymm"].astype(int)).copy()
    crsp = crsp.assign(yyyymm=crsp["yyyymm"].astype(int)).copy()

    df = raw.merge(crsp[["yyyymm", "vol_ann"]], on="yyyymm", how="inner")
    df = df[(df["yyyymm"] >= 195206) & (df["yyyymm"] <= 202203)].copy()
    # Centered 4-quarter moving averages (same smoothing choice as legacy code).
    df = df.assign(
        date_q=quarter_date_from_yyyymm(df["yyyymm"]),
        flow_smooth=df["flow"].rolling(window=4, center=True, min_periods=4).mean(),
        vol_smooth=df["vol_ann"].rolling(window=4, center=True, min_periods=4).mean(),
    ).copy()
    df = df.assign(multiplier=(df["vol_smooth"] / np.sqrt(4.0)) / df["flow_smooth"]).copy()

    # Preserve only rows with fully defined smoothed series for plotting.
    return df.dropna(subset=["flow_smooth", "vol_smooth", "multiplier"]).copy()


def plot_figure(df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left panel: flows + volatility.
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(df["date_q"], df["flow_smooth"], color="tab:blue", lw=2.0, label="Flow")
    ax2.plot(
        df["date_q"],
        df["vol_smooth"],
        color="tomato",
        lw=2.0,
        label="Return volatility",
    )

    # Recession bars.
    rec = df["recession"].fillna(0).astype(int).to_numpy()
    for i in range(1, len(df)):
        if rec[i] == 1:
            ax.axvspan(df["date_q"].iloc[i - 1], df["date_q"].iloc[i], color="0.8", alpha=0.6)

    ax.axhline(0.0, color="black", lw=1.0, ls=(0, (1.5, 4.0)))
    ax.set_title("Quarterly flows and return volatility")
    ax.set_ylabel("Quarterly flow")
    ax2.set_ylabel("Return volatility (annualized)")
    ax.set_ylim(0.0, 0.05)
    ax2.set_ylim(0.0, 0.50)

    # Right panel: volatility multiplier.
    axm = axes[1]
    axm.plot(df["date_q"], df["multiplier"], color="cornflowerblue", lw=2.0)
    for i in range(1, len(df)):
        if rec[i] == 1:
            axm.axvspan(df["date_q"].iloc[i - 1], df["date_q"].iloc[i], color="0.8", alpha=0.6)
    axm.axhline(df["multiplier"].mean(), color="black", lw=1.2, ls=(0, (6.0, 6.0)))
    axm.axhline(0.0, color="black", lw=1.0, ls=(0, (1.5, 4.0)))
    axm.set_title("Volatility multiplier")
    axm.set_ylabel("Volatility / flow")

    OUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    df = prep_data()
    cols = [
        "date_q",
        "yyyymm",
        "flow",
        "flow_smooth",
        "vol_ann",
        "vol_smooth",
        "multiplier",
        "recession",
    ]
    df[cols].to_csv(OUT_DATA_PATH, index=False)
    plot_figure(df)
    print(f"Wrote data: {OUT_DATA_PATH} ({len(df)} rows)")
    print(f"Wrote figure: {OUT_FIG_PATH}")
    print(
        "Sample:",
        df["date_q"].iloc[0].strftime("%Y-%m"),
        "to",
        df["date_q"].iloc[-1].strftime("%Y-%m"),
    )
    print("Mean multiplier:", round(df["multiplier"].mean(), 3))


if __name__ == "__main__":
    main()
