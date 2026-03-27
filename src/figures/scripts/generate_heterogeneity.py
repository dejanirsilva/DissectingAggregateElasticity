import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
# Common (as before)
gamma_sigma2 = 0.04    # γ * ||σ||^2
y_star       = 0.03    # y^*
psi          = 2.0

# Heterogeneity setup
gamma0 = 30.0          # passive investors' risk aversion
gamma1 = 6.0
gamma2 = 2.0
w1     = 0.75          # wealth share of type 1 among active investors

# Denominator γ (user-specified)
gamma_baseline = 25.0

# Passive portfolio shares for each curve
alpha_p_hetero = 0.25  # for heterogeneous-preferences curve
alpha_p_nohet  = 0.45  # for homogeneous-preferences curve

# x_a grid
xa = np.linspace(0.05, 0.25, 700)

# Vector-friendly text embedding (nice PDF export)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# ----------------------------
# Formulas
# ----------------------------
# Prefactor: (1 - ψ^{-1}) * (γ||σ||^2 / y*)
const = (1.0 - 1.0/psi) * (gamma_sigma2 / y_star)

# Active-side wealth-weighted average risk aversion
E_u_gamma = w1 * gamma1 + (1.0 - w1) * gamma2

# Inverse market elasticity with heterogeneity:
# ε_M^{-1} = const * [ (1 - α_p)/x_a - (γ0 - E^u[γ_j]) / γ ]
hetero = const * ((1.0 - alpha_p_hetero) / xa - (gamma0 - E_u_gamma) / gamma_baseline)

# Homogeneous-preferences benchmark (remove γ-term):
# ε_M^{-1} = const * (1 - α_p) / x_a
nohet  = const * ((1.0 - alpha_p_nohet) / xa)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(7.5, 5.5))
plt.plot(xa, hetero, label="Heterogeneous preferences", color="tab:blue",  lw=3.2)
plt.plot(xa, nohet,  label="Homogeneous preferences",   color="tab:orange", lw=3.2, ls="--")

plt.xlabel(r"Wealth share of active investors, $x_a$")
plt.ylabel(r"Inverse market elasticity, $\varepsilon_M^{-1}$")
plt.title("Price Impact")
plt.legend(frameon=False)

# Y-limits with a bit of padding
ymin = min(hetero.min(), nohet.min())
ymax = max(hetero.max(), nohet.max())
pad  = 0.07 * (ymax - ymin if ymax > ymin else 1.0)
plt.ylim(ymin - pad, ymax + pad)
plt.xlim(xa.min(), xa.max())

plt.grid(True, linestyle=":", linewidth=1)
plt.tight_layout()

# Save
plt.savefig("price_impact_w1_hetero_vs_nohet_labels.png", dpi=300, bbox_inches="tight")
plt.savefig("price_impact_w1_hetero_vs_nohet_labels.pdf", bbox_inches="tight")
plt.show()