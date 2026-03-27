# Minimal driver to generate plots and outputs
include(joinpath(@__DIR__, "perturbation.jl"))

# Plot 1: 1st and 2nd order Taylor series vs Closed-form
p1 = PerturbationPlots.generate_plot1()

# Plot 2: Taylor series orders 0 through 5: gammahat1 = 0.3, gammahat2 = -0.3
p2 = PerturbationPlots.generate_plot2(gammahat1=0.3, gammahat2=-0.3)

# Plot 3: Taylor series orders 0 through 5: gammahat1 = 0.45, gammahat2 = -0.45
p3 = PerturbationPlots.generate_plot2(gammahat1=0.45, gammahat2=-0.45)

# Composite vs Outer plots (tunable δ_scale, y_floor)
pc_base = PerturbationPlots.generate_composite_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.3, γhat2=0.3), δ_scale=100., y_floor=1e-6)

pc_alt = PerturbationPlots.generate_composite_plot(p=PerturbationLocalTaylor.Params(γhat1=0.45, γhat2=-0.45), δ_scale=10., y_floor=1e-6)

# Residual plots (log10 |residual|): Orders 0–5
p_base = PerturbationPlots.generate_residuals_by_order_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.3, γhat2=0.3))

p_alt = PerturbationPlots.generate_residuals_by_order_plot(p=PerturbationLocalTaylor.Params(γhat1=0.45, γhat2=-0.45))

# Composite residual plots (log10 |residual|)
pr_comp_base = PerturbationPlots.generate_residuals_composite_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.3, γhat2=0.3), δ_scale=0.2, y_floor=1e-6)

pr_comp_alt = PerturbationPlots.generate_residuals_composite_plot(p=PerturbationLocalTaylor.Params(γhat1=0.45, γhat2=-0.45), δ_scale=0.7, y_floor=1e-6)

# σx plots for regular perturbation (orders 0–5, baseline γ̂)
psx = PerturbationPlots.generate_sigmax_orders_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.75, γhat2=0.75))

# c1x, c2x plots for regular perturbation (orders 0–5, baseline γ̂)
pcx = PerturbationPlots.generate_cx_orders_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.75, γhat2=0.75))

a = PerturbationPlots.generate_weighted_cx_orders_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.75, γhat2=0.75))
# x^alpha * c2x (set alpha as needed)
p_c2x_power = PerturbationPlots.generate_weighted_c2x_power_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.75, γhat2=0.75), alpha=0.0)

# yx across orders
pyx = PerturbationPlots.generate_yx_orders_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.75, γhat2=0.75))

# B(x) = x*yx/y across orders
pB = PerturbationPlots.generate_B_orders_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.95, γhat2=0.95, κhat=0.0))

# B1, B2 across orders
pB12 = PerturbationPlots.generate_B1B2_orders_plot(p=PerturbationLocalTaylor.Params(γhat1=-0.95, γhat2=0.9, κhat=0.0))





