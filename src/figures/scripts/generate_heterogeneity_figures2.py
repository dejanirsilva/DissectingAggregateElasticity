import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
gamma_sigma2 = 0.04   # γ * ||σ||^2
y_star       = 0.03   # y^*
psi          = 2.0
gamma_den    = 5.0    # denominator γ

gamma0 = 30.0         # passive
gamma1 = 27.5         # active type 1
gamma2 = 2.0          # active type 2

alpha_p = 0.25        # fixed passive portfolio share

# Vector-friendly text embedding (nice PDF export)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# Prefactor: (1 - ψ^{-1}) * (γ||σ||^2 / y*)
const = (1.0 - 1.0/psi) * (gamma_sigma2 / y_star)

# Grid for w1 ∈ [0,1]
w1 = np.linspace(0.0, 1.0, 800)

def inv_elast_w1_xa(w1, xa):
    """ε_M^{-1}(w1; xa) with α_p fixed."""
    E_u = w1 * gamma1 + (1.0 - w1) * gamma2
    return const * ((1.0 - alpha_p) / xa - (gamma0 - E_u) / gamma_den)

# Curves: xa = 0.050 (dashed), 0.075 (solid), 0.100 (dash-dot)
xa_values   = [0.050, 0.075, 0.100]
line_styles = {0.050: '--', 0.075: '-', 0.100: '-.'}
colors      = {0.050: 'tab:orange', 0.075: 'tab:blue', 0.100: 'tab:green'}

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(7.5, 5.2))
ys_all = []
for xa in xa_values:
    yvals = inv_elast_w1_xa(w1, xa)
    ys_all.append(yvals)
    plt.plot(
        w1, yvals,
        linestyle=line_styles[xa],
        color=colors[xa],
        linewidth=3.0,
        label=rf"$x_a = {xa:.3f}$"
    )

plt.xlabel(r"Share of type 1 among actives, $w_1$")
plt.ylabel(r"Inverse market elasticity, $\varepsilon_M^{-1}$")
plt.axhline(0, linestyle=":", linewidth=1)
plt.legend(frameon=False, loc="best")
plt.xlim(0, 1)

# y-limits with padding
ys_all = np.vstack(ys_all)
ymin, ymax = ys_all.min(), ys_all.max()
pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
plt.ylim(ymin - pad, ymax + pad)

plt.grid(True, linestyle=":", linewidth=1)
plt.tight_layout()

# Save
plt.savefig("price_impact_vs_w1_by_xa_ap025_xa050_075_100_SOLID075.png", dpi=300, bbox_inches="tight")
plt.savefig("price_impact_vs_w1_by_xa_ap025_xa050_075_100_SOLID075.pdf", bbox_inches="tight")
plt.show()