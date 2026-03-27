#!/usr/bin/env python3
"""Stylized return-predictability scatter for the paper (replace with simulation export when available)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match Table tab:return_predictability slopes (illustrative sample)
SLOPE_FULL = -0.49
SLOPE_BAD = -0.66
RNG = np.random.default_rng(0)

n_full = 1200
n_bad = 350

# log(P/D): full sample spread; "bad" = low PD tail
x_full = RNG.normal(0.35, 0.45, n_full)
x_bad = RNG.normal(-0.85, 0.22, n_bad)

intercept_full = 0.25
intercept_bad = 0.05

noise_full = RNG.normal(0, 0.35, n_full)
noise_bad = RNG.normal(0, 0.4, n_bad)

y_full = intercept_full + SLOPE_FULL * x_full + noise_full
y_bad = intercept_bad + SLOPE_BAD * x_bad + noise_bad

xf = np.linspace(x_full.min(), x_full.max(), 100)
xb = np.linspace(x_bad.min(), x_bad.max(), 100)

fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=220)
ax.scatter(x_full, y_full, s=4, c="#1f77b4", alpha=0.35, linewidths=0, label="Normal state")
ax.scatter(x_bad, y_bad, s=5, c="#d62728", alpha=0.45, linewidths=0, label="Bad state")
ax.plot(xf, intercept_full + SLOPE_FULL * xf, color="#1f77b4", lw=1.8)
ax.plot(xb, intercept_bad + SLOPE_BAD * xb, color="#d62728", lw=1.8)
ax.set_xlabel(r"$\log(P_t/D_t)$")
ax.set_ylabel("Cumulative ex. ret.")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
fig.tight_layout()

out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(out_dir, "return_predictability.png")
fig.savefig(out_path)
print("wrote", out_path)
