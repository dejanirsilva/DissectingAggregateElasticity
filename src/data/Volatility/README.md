# Figure 6 Replication (Quarterly Flows and Return Volatility)

This folder contains the minimal pipeline to replicate Figure 6 from scratch using:

- `raw_data.csv` for the quarterly flow series (processed FoF series used in the paper),
- CRSP daily value-weighted index returns (`crsp.dsi.vwretd`) for realized volatility.

## Files Kept

- `raw_data.csv`: Quarterly flow input used by the paper workflow.
- `compute_crsp_quarterly_vol.py`: Pulls CRSP daily data from WRDS and computes quarterly realized volatility.
- `replicate_figure6.py`: Merges flow + volatility, applies smoothing, and generates the figure.

## Volatility Definition

For each quarter `q`, with daily return `r_d`:

- Quarterly realized variance: `RV_q = sum_{d in q} r_d^2`
- Quarterly realized volatility (annualized): `vol_ann_q = sqrt(RV_q) * sqrt(4)`

This matches the realized-volatility approach used in the replication pipeline.

## Prerequisites

1. WRDS access with CRSP permissions.
2. A valid `~/.pgpass` entry for WRDS.
3. Python packages: `wrds`, `pandas`, `numpy`, `matplotlib`.

## Reproduce

Run from repository root:

```bash
python3 src/data/Volatility/compute_crsp_quarterly_vol.py \
  --output src/data/Volatility/crsp_quarterly_realized_vol.csv

python3 src/data/Volatility/replicate_figure6.py
```

## Outputs

The second script writes:

- `src/data/Volatility/figure6_rebuilt_data.csv`
- `src/data/Volatility/figs/figure6_from_scratch.png`

The left-panel axis limits are fixed to match the original figure scale:

- Flow axis: `0` to `0.05`
- Volatility axis: `0` to `0.50`
