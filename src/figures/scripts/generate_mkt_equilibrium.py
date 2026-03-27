import matplotlib.pyplot as plt
import numpy as np

# Vector-friendly text embedding
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# Slopes (linearized)
m_blue, m_orng  = -1.2,  1.0   # goods market lines
m_green, m_red  = -0.9,  0.9   # risky-asset lines

# Domains
c = np.linspace(-1, 1, 200)
q = np.linspace(-1, 1, 200)

# Baseline lines (through (0,0))
p_blue  = m_blue * c
p_orng  = m_orng * c
p_green = m_green * q
p_red   = m_red   * q

# Comparative statics: LEFT shift (meet at q = -0.50, p = 0)
delta = 0.50
p_green_shift = m_green * (q + delta)  # shift left
p_red_shift   = m_red   * (q + delta)  # shift left
q_meet, p_meet = -delta, 0.0

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# (1) Risky asset (left) — green/red, dashed left shift
ax = axes[0]
ax.plot(q, p_green,        label="active demand", color="green", lw=2.5)
ax.plot(q, p_red,          label="net supply",    color="red",   lw=2.5)
ax.plot(q, p_green_shift, '--', color="green", lw=2.5)
ax.plot(q, p_red_shift,   '--', color="red",   lw=2.5)
ax.plot([0, q_meet], [0, p_meet], 'ko', markersize=5)  # dots at (0,0) and (-0.5,0)
ax.axhline(0, color="gray", linestyle=":", lw=1)
ax.axvline(0, color="gray", linestyle=":", lw=1)
ax.set_xlim(-1,1); ax.set_ylim(-1,1)
ax.set_title("Market for risky asset")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$\hat{p}$")
ax.legend(frameon=False)
ax.tick_params(axis='x', pad=8)
ax.tick_params(axis='y', pad=8)

# (2) Goods (right) — blue/orange, clean panel
ax = axes[1]
ax.plot(c, p_blue,  label="div. yield", color="blue",   lw=2.5)
ax.plot(c, p_orng,  label="avg. c/w ratio",     color="orange", lw=2.5)
ax.plot(0, 0, 'ko', markersize=5)
ax.axhline(0, color="gray", linestyle=":", lw=1)
ax.axvline(0, color="gray", linestyle=":", lw=1)
ax.set_xlim(-1,1); ax.set_ylim(-1,1)
ax.set_title("Market for goods")
ax.set_xlabel(r"$c,\, y$")
ax.set_ylabel(r"$\hat{p}$")
ax.legend(frameon=False)
ax.tick_params(axis='x', pad=8)
ax.tick_params(axis='y', pad=8)

plt.tight_layout()
plt.savefig("mkt_equilibrium.png", dpi=300, bbox_inches="tight")
# plt.savefig("mkt_equilibrium.pdf", bbox_inches="tight")
plt.show()
