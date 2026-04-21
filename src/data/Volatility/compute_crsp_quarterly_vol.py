#!/usr/bin/env python3
"""
Compute quarterly realized return volatility from CRSP daily index returns.

Definition used:
    rv_q = sqrt(sum_{d in quarter} r_d^2)
    vol_ann = rv_q * sqrt(4)

This script queries CRSP daily stock index returns (`crsp.dsi.vwretd`) via WRDS
and writes a quarterly series aligned with the paper sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import wrds


def load_wrds_credentials_from_pgpass(pgpass_path: Path) -> dict[str, str | int]:
    line = pgpass_path.read_text().strip().splitlines()[0]
    host, port, dbname, user, pwd = line.split(":", 4)
    return {
        "wrds_hostname": host,
        "wrds_port": int(port),
        "wrds_dbname": dbname,
        "wrds_username": user,
        "wrds_password": pwd,
    }


def compute_quarterly_vol(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.assign(quarter=df["date"].dt.to_period("Q"))
        .groupby("quarter", as_index=False)
        .agg(
            n_daily=("vwretd", "size"),
            rv_q=("vwretd", lambda x: (x.pow(2).sum()) ** 0.5),
        )
        .copy()
    )
    out = out.assign(
        vol_ann=out["rv_q"] * (4**0.5),
        yyyymm=out["quarter"].dt.qyear * 100 + out["quarter"].dt.quarter * 3,
    )
    return out[["quarter", "yyyymm", "n_daily", "rv_q", "vol_ann"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="1952-04-01")
    parser.add_argument("--end", default="2022-03-31")
    parser.add_argument(
        "--output",
        default="src/data/Volatility/crsp_quarterly_realized_vol.csv",
    )
    parser.add_argument("--pgpass", default=str(Path.home() / ".pgpass"))
    args = parser.parse_args()

    creds = load_wrds_credentials_from_pgpass(Path(args.pgpass))
    conn = wrds.Connection(verbose=False, **creds)
    try:
        query = f"""
        select date, vwretd
        from crsp.dsi
        where date between '{args.start}' and '{args.end}'
        order by date
        """
        daily = conn.raw_sql(query).copy()
    finally:
        conn.close()

    daily = daily.assign(
        date=pd.to_datetime(daily["date"]),
        vwretd=pd.to_numeric(daily["vwretd"], errors="coerce"),
    ).copy()
    daily = daily.dropna(subset=["vwretd"]).copy()

    quarterly = compute_quarterly_vol(daily)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quarterly.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(quarterly)} rows)")
    print(
        f"Sample: {quarterly['quarter'].iloc[0]} to {quarterly['quarter'].iloc[-1]}"
    )
    print(quarterly.head(3).to_string(index=False))
    print(quarterly.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
