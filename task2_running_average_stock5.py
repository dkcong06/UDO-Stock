"""Task 2: stock 5, phi in {0.7,0.5,0.3}; mirrors exchange_analysis eqs. 3-6."""
import numpy as np
import matplotlib.pyplot as plt

p = np.genfromtxt("stock_prices1.csv", delimiter=",")
if p.ndim == 1:
    p = p.reshape(1, -1)
if np.isnan(p[:, -1]).all():
    p = p[:, :-1]
p = p[:, 4].astype(float)
n = len(p)
t = np.arange(1, n + 1)
fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

for phi, c in zip((0.7, 0.5, 0.3), ("C0", "C1", "C2")):
    phi = float(np.clip(phi, 0.0, 1.0))
    sp, sv, sa, va = p.copy(), np.zeros(n), np.zeros(n), np.zeros(n)
    for day in range(1, n - 1):
        d, dp, dm = day, day + 1, day - 1
        sp[dp] = (1 - phi) * sp[d] + phi * p[dp]
        den = sp[dp] + sp[dm]
        den = den if den != 0 else 1e-10
        sv[dp] = (sp[dp] - sp[dm]) / den
        sa[dp] = (sv[dp] - sv[dm]) / 2.0
        va[dp] = (1 - phi) * va[d] + phi * sv[dp] ** 2
    ax[0].plot(t, sp, c, label=f"$\\phi={phi}$")
    ax[1].plot(t, sv, c)
    ax[2].plot(t, sa, c)
    ax[3].plot(t, np.sqrt(np.maximum(va, 0.0)), c)

ax[0].set_ylabel(r"$\bar{p}$")
ax[0].legend(fontsize=8)
ax[1].set_ylabel(r"$\dot{\bar{p}}$")
ax[2].set_ylabel(r"$\ddot{\bar{p}}$")
ax[3].set_ylabel(r"$\sqrt{\bar{v}^2}$")
ax[3].set_xlabel("day")
for a in ax:
    a.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()
