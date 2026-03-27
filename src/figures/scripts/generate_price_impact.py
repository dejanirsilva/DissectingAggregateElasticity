import numpy as np
import matplotlib.pyplot as plt

γ = 25
σ = 0.04
γ * σ**2

ω0 =0.80
ω1 = 0.15
ω2 = 1- ω0 - ω1
γ0 = 30.0
γ1 = 6.0
γ2 = 2.0
γ = ω0 * γ0 + ω1 * γ1 + ω2 * γ2

(30 - (ω1 * γ1 + ω2 * γ2)/(ω1+ω2))/γ
ω1/(ω1+ω2)
(ω1 * γ1 + ω2 * γ2)/(ω1+ω2)



# --- Parameters ---
gamma_sigma2 = 0.04   # γ * ||σ||^2
y_star = 0.03         # y^*
psi = 2.0
alpha_ps = [0.00, 0.25, 0.50, 1.00]

# Vector-friendly text embedding (nice PDF export)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42

# Constant multiplier: (1 - ψ^{-1}) * (γ||σ||^2 / y*)
const = (1.0 - 1.0/psi) * (gamma_sigma2 / y_star)

# x_a grid (start at 0.05, end at 0.5)
xa = np.linspace(0.05, 0.5, 700)

def inv_elasticity(xa, alpha_p, const):
    """Inverse market elasticity: ε_M^{-1} = const * (1 - α_p) / x_a"""
    return const * (1.0 - alpha_p) / xa

# Line styles: only α_p=0.25 is solid; others dotted/dashed/dash-dot
line_styles = {
    0.00: ':',    # dotted
    0.25: '-',    # solid
    0.50: '--',   # dashed
    1.00: '-.'    # dash-dot
}

# Colors (α_p=1.00 in purple for contrast)
colors = {
    0.00: 'tab:orange',
    0.25: 'tab:blue',
    0.50: 'tab:green',
    1.00: 'purple'
}

# ---- Plot ----
plt.figure(figsize=(7.5, 5.5))
for ap in alpha_ps:
    plt.plot(
        xa,
        inv_elasticity(xa, ap, const),
        linestyle=line_styles[ap],
        color=colors[ap],
        linewidth=3.2,
        label=fr"$\bar{{\alpha}}_p = {ap:.2f}$"
    )

plt.xlabel(r"Wealth share of active investors, $x_a$")
plt.ylabel(r"Inverse market elasticity, $\varepsilon_M^{-1}$")
plt.title("Price Impact")
plt.legend(frameon=False)

# y-axis: a bit below zero so the α_p = 1.00 line (at 0) is visible
ymax = const * (1.0 - min(alpha_ps)) / xa.min() * 1.05
plt.ylim(-0.2, ymax)
plt.xlim(xa.min(), xa.max())

plt.grid(True, linestyle=":", linewidth=1)
plt.tight_layout()

# Save (PNG + PDF)
plt.savefig("inv_market_elasticity_vs_xa_title_PRICE_IMPACT_v3.png", dpi=300, bbox_inches="tight")
plt.savefig("inv_market_elasticity_vs_xa_title_PRICE_IMPACT_v3.pdf", bbox_inches="tight")
plt.show()
