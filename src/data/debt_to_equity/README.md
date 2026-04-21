# Debt-to-Equity Data Pipeline

This folder contains a minimal pipeline to:

1. Download two US Flow of Funds leverage series from FRED.
2. Compute averages since 1980 (default).

## Series

- `NCBCMDPMVCE` (`FL104104016.Q`): Debt as % of market value of corporate equities.
- `NCBCMDPNWHC` (`FL104104026.Q`): Debt as % of net worth (historical cost; book-style leverage).

## Run

From the repository root:

```bash
python3 src/data/debt_to_equity/run_pipeline.py
```

Optional custom start year:

```bash
python3 src/data/debt_to_equity/run_pipeline.py --start-year 1990
```

## Outputs

- `src/data/debt_to_equity/raw/NCBCMDPMVCE.csv`: cleaned quarterly market-value series.
- `src/data/debt_to_equity/raw/NCBCMDPNWHC.csv`: cleaned quarterly historical-cost (book) series.
- `src/data/debt_to_equity/processed/average_since_1980.csv`: summary table
  with one row per series, sample window, number of observations, average
  percent, and average ratio (percent divided by 100).
