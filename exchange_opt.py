from exchange_analysis_v3 import exchange_analysis
import numpy as np
from scipy.optimize import minimize, dual_annealing

stock_prices = np.genfromtxt("stock_prices1.csv", delimiter=",")  # DO NOT CHANGE
if np.isnan(stock_prices[:, -1]).all():
    stock_prices = stock_prices[:, :-1]
constants = [0, stock_prices]  # DO NOT CHANGE

PENALTY = 1e6


def objective(x):
    """added to ensure that Buy threshold is always greater than Sell threshold"""
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
    (-1, 1),      # B 
    (-1, 1),      # S
]
"""Used dual annealing with each initial guess to find rough optima
"""
initial_guesses = [
    [-10,  2,  -2, 0.9,  0.5,  0.1,  -0.1],   # Guess 1 (PDF example)
    [ -5,  5,  -5, 0.8,  0.3,  0.05, -0.05],   # Guess 2
    [ -8,  3,  -3, 0.7,  0.4,  0.15, -0.15],   # Guess 3
    [-12,  4,  -4, 0.85, 0.6,  0.08, -0.08],   # Guess 4
    [ -6, -6,  7, 0.83, 0.033,0.1,  -0.1]    # Guess 5
]

da_results = []
for i, x0 in enumerate(initial_guesses):
    da = dual_annealing(
        objective, bounds, seed=42 + i, maxiter=500, x0=x0,
    )
    da_results.append(da)

# Collect all DA results as seeds for local polish
polish_guesses = [da.x for da in da_results]

best_cost = np.inf
best_x = None

for i, x0 in enumerate(polish_guesses):
    for method in ["Powell", "Nelder-Mead"]:
        res = minimize(objective, x0, method=method,
                       options={"maxiter": 10000})
        if res.fun < PENALTY and res.x[5] > res.x[6]:
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x.copy()
                x0 = res.x

# Sensitivity analysis
labels = ["q1", "q2", "q3", "fc", "phi", "B", "S"]
base_val = -best_cost

with open("optimization_results.txt", "w") as f:
    f.write("DUAL ANNEALING RESULTS\n")
    for i, (x0, da) in enumerate(zip(initial_guesses, da_results)):
        f.write(f"Guess {i+1}: {np.round(x0, 4)}\n")
        f.write(f"  DA optimal: {np.round(da.x, 4)}\n")
        f.write(f"  Value: ${-da.fun:.2f}  B={da.x[5]:.4f} S={da.x[6]:.4f}\n\n")

    f.write("LOCAL POLISH RESULTS\n")
    for i, x0_da in enumerate(polish_guesses):
        for method in ["Powell", "Nelder-Mead"]:
            res = minimize(objective, x0_da, method=method,
                           options={"maxiter": 10000})
            if res.fun < PENALTY and res.x[5] > res.x[6]:
                f.write(f"DA Guess {i+1} ({method}):\n")
                f.write(f"  Start:   {np.round(x0_da, 4)}\n")
                f.write(f"  Optimal: {np.round(res.x, 4)}\n")
                f.write(f"  Value: ${-res.fun:.2f}  B={res.x[5]:.4f} S={res.x[6]:.4f}\n\n")


    f.write("BEST RESULT\n")
    f.write(f"Parameters: {np.round(best_x, 6)}\n")
    f.write(f"  q1={best_x[0]:.4f}  q2={best_x[1]:.4f}  q3={best_x[2]:.4f}\n")
    f.write(f"  fc={best_x[3]:.4f}  phi={best_x[4]:.4f}\n")
    f.write(f"  B={best_x[5]:.4f}   S={best_x[6]:.4f}   (B > S: {best_x[5] > best_x[6]})\n")
    f.write(f"Full value (200 days): ${-best_cost:.2f}\n\n")


    f.write("SENSITIVITY ANALYSIS (+5% perturbation)\n")
    f.write(f"{'param':<6} {'optimal':>10} {'perturbed':>10} {'value($)':>10} {'change($)':>10}\n")
    for j in range(len(best_x)):
        x_pert = best_x.copy()
        x_pert[j] *= 1.05
        if x_pert[5] <= x_pert[6]:
            pert_val = 0.0
        else:
            pert_cost, _ = exchange_analysis(x_pert, constants)
            pert_val = -pert_cost
        f.write(f"{labels[j]:<6} {best_x[j]:>10.4f} {x_pert[j]:>10.4f} {pert_val:>10.2f} {pert_val - base_val:>+10.2f}\n")


exchange_analysis(best_x, [20, stock_prices])


