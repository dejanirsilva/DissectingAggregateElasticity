import matplotlib.pyplot as plt
import numpy as np

# ------- Parameters you might tweak -------
delta_q  = 0.50   # left shift of risky-asset dashed lines (x-direction)
p_target = 0.25   # new equilibrium price level after the shock (both panels)

# Vector-friendly text embedding
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# -------- Slopes (linearized) ----------
# Goods market: blue (div. yield) is downward; orange (avg. c/w) is upward
m_blue, m_orng  = -1.2,  1.0
# Risky-asset market: green (active demand) downward; red (net supply) upward
m_green, m_red  = -0.9,  0.9

# Domains
c = np.linspace(-1, 1, 200)
q = np.linspace(-1, 1, 200)

# Baseline lines (through (0,0))
p_blue  = m_blue * c    # dividend yield (downward, blue)
p_orng  = m_orng * c    # avg c/w ratio (upward, orange)
p_green = m_green * q   # active demand (downward, green)
p_red   = m_red   * q   # net supply (upward, red)

# ---------- Risky-asset panel: dashed curves shift LEFT and UP ----------
# New dashed lines: shift left by delta_q and vertically so they meet at (q=-delta_q, p=p_target)
p_green_shift = m_green * (q + delta_q) + p_target
p_red_shift   = m_red   * (q + delta_q) + p_target
q_meet_left, p_meet_left = -delta_q, p_target  # new intersection (left panel)

# ---------- Goods panel: shift ORANGE (avg c/w) upward to meet BLUE at p_target ----------
# Blue line stays fixed. Choose intercept for orange so intersection occurs at p_target.
# Intersection solves: m_blue*c = m_orng*c + b  => c = b/(m_blue - m_orng)
# Then p_target = m_orng*c + b = b*m_blue/(m_blue - m_orng) => b = p_target*(m_blue - m_orng)/m_blue
b_orng = p_target * (m_blue - m_orng) / m_blue
p_orng_shift = m_orng * c + b_orng

# Intersection coordinate in goods panel (by construction):
c_meet_right = b_orng / (m_blue - m_orng)
p_meet_right = p_target

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (1) Market for risky asset (left)
ax = axes[0]
ax.plot(q, p_green,        label="active demand", color="green", lw=2.8)
ax.plot(q, p_red,          label="net supply",    color="red",   lw=2.8)
ax.plot(q, p_green_shift, '--', color="green", lw=2.8)
ax.plot(q, p_red_shift,   '--', color="red",   lw=2.8)
ax.plot([0], [0], 'ko', markersize=5)                 # initial equilibrium
ax.plot([q_meet_left], [p_meet_left], 'ko', markersize=5)  # new equilibrium
ax.axhline(0, color="gray", linestyle=":", lw=1)
ax.axvline(0, color="gray", linestyle=":", lw=1)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
ax.set_title("Market for risky asset")
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$\hat{p}$")
ax.legend(frameon=False)
ax.tick_params(axis='x', pad=8); ax.tick_params(axis='y', pad=8)

# (2) Market for goods (right)
ax = axes[1]
ax.plot(c, p_blue,        label="div. yield",      color="blue",   lw=2.8)   # fixed downward
ax.plot(c, p_orng,        label="avg. c/w ratio",  color="orange", lw=2.8)   # baseline upward
ax.plot(c, p_orng_shift, '--', color="orange", lw=2.8)                          # shifted upward
ax.plot([0], [0], 'ko', markersize=5)                       # initial equilibrium
ax.plot([c_meet_right], [p_meet_right], 'ko', markersize=5) # new equilibrium
ax.axhline(0, color="gray", linestyle=":", lw=1)
ax.axvline(0, color="gray", linestyle=":", lw=1)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
ax.set_title("Market for goods")
ax.set_xlabel(r"$c,\, y$")
ax.set_ylabel(r"$\hat{p}$")
ax.legend(frameon=False)
ax.tick_params(axis='x', pad=8); ax.tick_params(axis='y', pad=8)

plt.tight_layout()
plt.savefig("mkt_equilibrium2.png", dpi=300, bbox_inches="tight")
# plt.savefig("mkt_equilibrium2.pdf", bbox_inches="tight")
plt.show()