from exchange_analysis_v3 import exchange_analysis
import numpy as np
from scipy.optimize import minimize, dual_annealing

stock_prices = np.genfromtxt("stock_prices1.csv", delimiter=",")  # DO NOT CHANGE
if np.isnan(stock_prices[:, -1]).all():
    stock_prices = stock_prices[:, :-1]
constants = [0, stock_prices]  # DO NOT CHANGE

PENALTY = 1e6


def objective(x):
    # Enforce B > S: design_var[5] = B, design_var[6] = S
    if x[5] <= x[6]:
        return PENALTY
    cost, _ = exchange_analysis(x, constants)
    return cost


# design_var: [q1, q2, q3, fc, phi, B, S]
bounds = [
    (-30, 30),    # q1
    (-30, 30),    # q2
    (-30, 30),    # q3
    (0.01, 1.0),  # fc
    (0.01, 0.99), # phi
    (-1, 1),      # B  (must be > S)
    (-1, 1),      # S
]

# ---------- Stage 1: Dual Annealing (global) ----------
print("=" * 70)
print("Stage 1: Dual Annealing (full 200 days, B > S enforced)")
print("=" * 70)

da_result = dual_annealing(
    objective, bounds, seed=42, maxiter=500, x0=[-10, 2, -2, 0.9, 0.5, 0.1, -0.1],
)
print(f"  DA best:  {np.round(da_result.x, 4)}")
print(f"  Value: ${-da_result.fun:.2f}")
print(f"  B={da_result.x[5]:.4f} > S={da_result.x[6]:.4f}: {da_result.x[5] > da_result.x[6]}")

# ---------- Stage 2: Powell + Nelder-Mead polish ----------
print("\n" + "=" * 70)
print("Stage 2: Local polish (Powell + Nelder-Mead, multiple starts)")
print("=" * 70)

initial_guesses = [
    da_result.x,
    [-10, 2, -2, 0.9, 0.5, 0.1, -0.1],
    [-5, 5, -5, 0.8, 0.3, 0.05, -0.05],
    [-8, 3, -3, 0.7, 0.4, 0.15, -0.15],
    [-12, 4, -4, 0.85, 0.6, 0.08, -0.08],
    [-6, -6, 27, 0.83, 0.033, 0.1, -0.1],
]

best_cost = np.inf
best_x = None

for i, x0 in enumerate(initial_guesses):
    tag = "DA result" if i == 0 else f"Guess {i}"
    for method in ["Powell", "Nelder-Mead"]:
        res = minimize(objective, x0, method=method,
                       options={"maxiter": 10000})
        if res.fun < PENALTY and res.x[5] > res.x[6]:
            val = -res.fun
            if i == 0 or method == "Powell":
                print(f"\n  {tag} ({method}): {np.round(np.asarray(x0), 4)}")
                print(f"    Optimal:  {np.round(res.x, 4)}")
                print(f"    Value: ${val:.2f}  B={res.x[5]:.4f} S={res.x[6]:.4f}")
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x.copy()
                x0 = res.x  # feed into next method

print("\n" + "=" * 70)
print("BEST RESULT")
print("=" * 70)
print(f"Parameters: {np.round(best_x, 6)}")
print(f"  q1={best_x[0]:.4f}  q2={best_x[1]:.4f}  q3={best_x[2]:.4f}")
print(f"  fc={best_x[3]:.4f}  phi={best_x[4]:.4f}")
print(f"  B={best_x[5]:.4f}   S={best_x[6]:.4f}   (B > S: {best_x[5] > best_x[6]})")
print(f"Full value (200 days): ${-best_cost:.2f}")

# Sensitivity analysis
print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS (+5% perturbation)")
print("=" * 70)
labels = ["q1", "q2", "q3", "fc", "phi", "B", "S"]
base_val = -best_cost
print(f"{'param':<6} {'optimal':>10} {'perturbed':>10} {'value($)':>10} {'change($)':>10}")
for j in range(len(best_x)):
    x_pert = best_x.copy()
    x_pert[j] *= 1.05
    if x_pert[5] <= x_pert[6]:
        pert_val = 0.0
    else:
        pert_cost, _ = exchange_analysis(x_pert, constants)
        pert_val = -pert_cost
    print(f"{labels[j]:<6} {best_x[j]:>10.4f} {x_pert[j]:>10.4f} {pert_val:>10.2f} {pert_val - base_val:>+10.2f}")

# Generate figures
print("\nGenerating figures with best parameters...")
exchange_analysis(best_x, [20, stock_prices])
