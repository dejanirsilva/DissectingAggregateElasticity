import os
from pathlib import Path


mpl_config_dir = Path("/private/tmp/matplotlib-dae")
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


# ------- Parameters you might tweak -------
p_target = 0.25  # common new equilibrium price in both panels

# Vector-friendly text embedding
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# -------- Slopes (linearized) ----------
# Goods market: blue (div. yield) is downward; orange (avg. c/w) is upward
m_blue, m_orng = -1.2, 1.0
# Risky-asset market: green (active demand) downward; red (net supply) upward
m_green, m_red = -0.9, 0.9

# Domains
c = np.linspace(-1, 1, 200)
q = np.linspace(-1, 1, 200)

# Baseline lines (through (0,0))
p_blue = m_blue * c  # dividend yield (downward, blue)
p_orng = m_orng * c  # avg c/w ratio (upward, orange)
p_green = m_green * q  # active demand (downward, green)
p_red = m_red * q  # net supply (upward, red)

# ---------- Goods panel: shift avg. c/w LEFT ----------
# A left shift by delta_cw gives p = m_orng * (c + delta_cw).
# Choose delta_cw so the shifted orange curve intersects the fixed blue curve
# at p_target.
c_meet_right = p_target / m_blue
delta_cw = c_meet_right - (p_target / m_orng)
p_orng_shift = m_orng * (c - delta_cw)
p_meet_right = p_target

# ---------- Risky-asset panel: shift active demand RIGHT ----------
# Keep net supply fixed. The new intersection must have the same price
# p_target, so q is pinned down by the red line.
q_meet_left = p_target / m_red
delta_active = q_meet_left - (p_target / m_green)
p_green_shift = m_green * (q - delta_active)
p_meet_left = p_target

# Verify the plotted shifts impose the same price in both panels.
assert np.isclose(m_blue * c_meet_right, p_target)
assert np.isclose(m_orng * (c_meet_right - delta_cw), p_target)
assert np.isclose(m_red * q_meet_left, p_target)
assert np.isclose(m_green * (q_meet_left - delta_active), p_target)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (1) Market for risky asset (left)
ax = axes[0]
ax.plot(q, p_green, label="active demand", color="green", lw=2.8)
ax.plot(q, p_red, label="net supply", color="red", lw=2.8)
ax.plot(q, p_green_shift, "--", color="green", lw=2.8)
ax.plot([0], [0], "ko", markersize=5)  # initial equilibrium
ax.plot([q_meet_left], [p_meet_left], "ko", markersize=5)  # new equilibrium
ax.axhline(0, color="gray", linestyle=":", lw=1)
ax.axvline(0, color="gray", linestyle=":", lw=1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_title("Market for risky asset")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$\hat{p}$")
ax.legend(frameon=False)
ax.tick_params(axis="x", pad=8)
ax.tick_params(axis="y", pad=8)

# (2) Market for goods (right)
ax = axes[1]
ax.plot(c, p_blue, label="div. yield", color="blue", lw=2.8)
ax.plot(c, p_orng, label="avg. c/w ratio", color="orange", lw=2.8)
ax.plot(c, p_orng_shift, "--", color="orange", lw=2.8)
ax.plot([0], [0], "ko", markersize=5)  # initial equilibrium
ax.plot([c_meet_right], [p_meet_right], "ko", markersize=5)  # new equilibrium
ax.axhline(0, color="gray", linestyle=":", lw=1)
ax.axvline(0, color="gray", linestyle=":", lw=1)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_title("Market for goods")
ax.set_xlabel(r"$c,\, y$")
ax.set_ylabel(r"$\hat{p}$")
ax.legend(frameon=False)
ax.tick_params(axis="x", pad=8)
ax.tick_params(axis="y", pad=8)

plt.tight_layout()

repo_root = Path(__file__).resolve().parents[3]
out_dir = repo_root / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "mkt_equilibrium_cw_left_active_right.png", dpi=300, bbox_inches="tight")
plt.savefig(out_dir / "mkt_equilibrium_cw_left_active_right.pdf", bbox_inches="tight")
