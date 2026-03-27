#!/usr/bin/env python3
"""
Example demonstrating the compare_derivative_methods() function.
This shows the differences between computing derivatives with respect to x vs s.
"""

# %% 
import numpy as np
import jax.numpy as jnp
from passive_investor import (
    Config, ModelParameters, 
    compute_derivatives, compute_derivatives_s_direct,
    compare_derivative_methods, test_s_derivatives_stability
)

# Set up a simple test case
config = Config(N=(5,), lower_bound=(1e-6,), upper_bound=(1.0-1e-6,), use_log_state=False, use_s_derivatives=True)
params = ModelParameters()

# Create some dummy Chebyshev coefficients for testing
# These represent the coefficients for wa and wp functions
θ = {
    'θa': jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # Coefficients for wa
    'θp': jnp.array([0.2, 0.1, 0.4, 0.3, 0.6])   # Coefficients for wp
}

print("=" * 60)
print("DERIVATIVE COMPARISON EXAMPLE")
print("=" * 60)
print()

# %%
# Test at different x values
test_x_values = jnp.array([0.01, 0.1, 0.5, 0.9, 0.99])

for i, x in enumerate(test_x_values):
    print(f"Testing at x = {x:.3f}")
    print("-" * 40)
    
    # Method 1: Direct x-derivatives (original method)
    dF_x = compute_derivatives(config, θ, params, x)
    
    # Method 2: s-derivatives converted to x-derivatives (new method)
    dF_s = compute_derivatives_s_direct(config, θ, params, x, return_x=True)
    
    # Compare the methods
    differences = compare_derivative_methods(config, θ, params, x)
    
    print("Direct x-derivatives:")
    print(f"  wax = {dF_x['wax']:.6f}")
    print(f"  wpx = {dF_x['wpx']:.6f}")
    print(f"  waxx = {dF_x['waxx']:.6f}")
    print(f"  wpxx = {dF_x['wpxx']:.6f}")
    print()
    
    print("s-derivatives converted to x-derivatives:")
    print(f"  wax = {dF_s['wax']:.6f}")
    print(f"  wpx = {dF_s['wpx']:.6f}")
    print(f"  waxx = {dF_s['waxx']:.6f}")
    print(f"  wpxx = {dF_s['wpxx']:.6f}")
    print()
    
    print("Differences (absolute):")
    for key in ['wax', 'wpx', 'waxx', 'wpxx']:
        abs_diff = differences[f'{key}_abs_diff']
        rel_diff = differences[f'{key}_rel_diff']
        print(f"  {key}: abs_diff = {abs_diff:.2e}, rel_diff = {rel_diff:.2e}")
    
    print()
    print("s-derivatives (computed directly):")
    print(f"  was = {dF_s['was']:.6f}")
    print(f"  wps = {dF_s['wps']:.6f}")
    print(f"  wass = {dF_s['wass']:.6f}")
    print(f"  wpss = {dF_s['wpss']:.6f}")
    print()
    print("=" * 60)
    print()

print("SUMMARY:")
print("The compare_derivative_methods() function shows that both methods")
print("should give identical results (up to numerical precision).")
print("The s-derivatives approach is more numerically stable, especially")
print("near boundaries where x is close to 0 or 1.")

# %%
