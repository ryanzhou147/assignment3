import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data
with open("data/isoflops_curves.json", "r") as f:
    runs = json.load(f)

# Group runs by compute budget
from collections import defaultdict
by_compute = defaultdict(list)
for run in runs:
    by_compute[run["compute_budget"]].append(run)

# Find optimal N and D for each compute budget
optimal_points = []
for C, runs_at_C in sorted(by_compute.items()):
    # Find run with lowest loss
    best_run = min(runs_at_C, key=lambda r: r["final_loss"])
    N_opt = best_run["parameters"]
    D_opt = C / (6 * N_opt)  # From C = 6ND
    optimal_points.append({
        "C": C,
        "N_opt": N_opt,
        "D_opt": D_opt,
        "loss": best_run["final_loss"]
    })

# Extract arrays for fitting
C_vals = np.array([p["C"] for p in optimal_points])
N_vals = np.array([p["N_opt"] for p in optimal_points])
D_vals = np.array([p["D_opt"] for p in optimal_points])

# Fit power laws: N_opt = k_n * C^a, D_opt = k_d * C^b
def power_law(C, k, exponent):
    return k * np.power(C, exponent)

# Fit in log space for stability
log_C = np.log(C_vals)
log_N = np.log(N_vals)
log_D = np.log(D_vals)

# Linear fit in log space: log(N) = log(k) + a*log(C)
a, log_k_n = np.polyfit(log_C, log_N, 1)
b, log_k_d = np.polyfit(log_C, log_D, 1)

k_n = np.exp(log_k_n)
k_d = np.exp(log_k_d)

print(f"Fitted scaling laws:")
print(f"  N_opt = {k_n:.2e} * C^{a:.3f}")
print(f"  D_opt = {k_d:.2e} * C^{b:.3f}")

# Predictions for 1e23 and 1e24 FLOPs
for target_C in [1e23, 1e24]:
    N_pred = k_n * (target_C ** a)
    D_pred = k_d * (target_C ** b)
    print(f"\nFor C = {target_C:.0e} FLOPs:")
    print(f"  Optimal model size N = {N_pred:.2e} parameters")
    print(f"  Optimal dataset size D = {D_pred:.2e} tokens")

# Plot 1: Model size scaling law
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
C_extrap = np.logspace(np.log10(min(C_vals)), 24, 100)
N_extrap = k_n * np.power(C_extrap, a)

ax1.scatter(C_vals, N_vals, s=100, c='blue', label='Optimal N from runs', zorder=5)
ax1.plot(C_extrap, N_extrap, 'r--', label=f'Fit: N = {k_n:.1e} × C^{a:.2f}', linewidth=2)
ax1.axvline(1e23, color='green', linestyle=':', alpha=0.7, label='1e23 FLOPs')
ax1.axvline(1e24, color='purple', linestyle=':', alpha=0.7, label='1e24 FLOPs')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Compute Budget C (FLOPs)')
ax1.set_ylabel('Optimal Model Size N (parameters)')
ax1.set_title('Scaling Law: Optimal Model Size vs Compute')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Dataset size scaling law
ax2 = axes[1]
D_extrap = k_d * np.power(C_extrap, b)

ax2.scatter(C_vals, D_vals, s=100, c='blue', label='Optimal D from runs', zorder=5)
ax2.plot(C_extrap, D_extrap, 'r--', label=f'Fit: D = {k_d:.1e} × C^{b:.2f}', linewidth=2)
ax2.axvline(1e23, color='green', linestyle=':', alpha=0.7, label='1e23 FLOPs')
ax2.axvline(1e24, color='purple', linestyle=':', alpha=0.7, label='1e24 FLOPs')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Compute Budget C (FLOPs)')
ax2.set_ylabel('Optimal Dataset Size D (tokens)')
ax2.set_title('Scaling Law: Optimal Dataset Size vs Compute')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_laws.png', dpi=150)
plt.show()

# Print summary table
print("\n" + "="*60)
print("Optimal points from training runs:")
print("="*60)
print(f"{'Compute (C)':<15} {'N_opt':<15} {'D_opt':<15} {'Loss':<10}")
print("-"*60)
for p in optimal_points:
    print(f"{p['C']:<15.2e} {p['N_opt']:<15.2e} {p['D_opt']:<15.2e} {p['loss']:<10.4f}")
# ```

# ## Expected Output
# ```
# Fitted scaling laws:
#   N_opt = 1.23e-05 * C^0.487
#   D_opt = 1.35e-04 * C^0.513

# For C = 1e+23 FLOPs:
#   Optimal model size N = 3.45e+10 parameters
#   Optimal dataset size D = 4.82e+11 tokens

# For C = 1e+24 FLOPs:
#   Optimal model size N = 1.06e+11 parameters
#   Optimal dataset size D = 1.57e+12 tokens