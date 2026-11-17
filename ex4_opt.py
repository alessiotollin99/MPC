"""
Ex. 4 – sensitivity study on the 2-state model (same OCP as ex4.py, linear slack).

A) initialization: same problem, different set_initial -> same optimum or not?
B) initial state: different Tfloor(0) -> how much does the first-hour drop change?
"""

import casadi as cd
import numpy as np
import matplotlib.pyplot as plt

# -------------model data (same as ex4.py)----------
UA_home = 200  # W/K air-outdoor
C_home = 2 * 10**6  # J/K
UA_floor = 500  # W/K air-floor
C_floor = 1 * 10**7  # J/K

dt = 3600  # s
nsteps = 24  # hours
hours = np.arange(nsteps)
Text = 3 + 4 * np.sin(2 * np.pi * (hours - 9) / 24)
prices = np.where(hours < 6, 0.12, np.where((hours >= 17) & (hours < 21), 0.40, 0.20))
rho = 0.2  # €/K violation cost (linear)

# -------------integrator (one hour)----------
x = cd.SX.sym("x", 2)  # [Thouse, Tfloor]
p = cd.SX.sym("p", 2)  # [Qth, Text]
dx0 = UA_floor / C_home * (x[1] - x[0]) + UA_home / C_home * (p[1] - x[0])
dx1 = p[0] / C_floor - UA_floor / C_floor * (x[1] - x[0])
F = cd.integrator("F", "cvodes", {"x": x, "p": p, "ode": cd.vertcat(dx0, dx1)}, 0, dt)


def simulate(Thouse0, Tfloor0, Qth):
    """Plant simulation with given heat input profile -> (Thouse, Tfloor), nsteps+1 values each."""
    X = np.zeros((2, nsteps + 1))
    X[:, 0] = [Thouse0, Tfloor0]
    for k in range(nsteps):
        X[:, k + 1] = np.asarray(F(x0=X[:, k], p=[Qth[k], Text[k]])["xf"]).ravel()
    return X[0], X[1]


def solve_ocp(Thouse0, Tfloor0, guess=None):
    """Same OCP as ex4.py. guess: dict(Thouse, Tfloor, Qth) for set_initial, None = CasADi default (zeros)."""
    opti = cd.Opti()
    Thouse = opti.variable(nsteps + 1)
    Tfloor = opti.variable(nsteps + 1)
    Qth = opti.variable(nsteps)
    s_low = opti.variable(nsteps)
    s_upp = opti.variable(nsteps)

    Tsupply = Tfloor[:-1] + 5
    COP = 0.45 * (Tsupply + 273.15) / (Tsupply - Text)
    P = Qth / COP
    energy_cost = cd.sum1(P * prices / 1000 * dt / 3600)
    opti.minimize(energy_cost + rho * cd.sum1(s_low) + rho * cd.sum1(s_upp))

    for k in range(nsteps):
        x_next = F(x0=cd.vertcat(Thouse[k], Tfloor[k]), p=cd.vertcat(Qth[k], Text[k]))["xf"]
        opti.subject_to(cd.vertcat(Thouse[k + 1], Tfloor[k + 1]) == x_next)

    opti.subject_to(opti.bounded(0, P, 3000))
    opti.subject_to(s_low >= 0)
    opti.subject_to(s_upp >= 0)
    opti.subject_to(Thouse[1:] >= 20 - s_low)
    opti.subject_to(Thouse[1:] <= 23 + s_upp)
    opti.subject_to(Qth >= 0)
    opti.subject_to(Thouse[0] == Thouse0)
    opti.subject_to(Tfloor[0] == Tfloor0)

    if guess is not None:
        opti.set_initial(Thouse, guess["Thouse"])
        opti.set_initial(Tfloor, guess["Tfloor"])
        opti.set_initial(Qth, guess["Qth"])

    opti.solver("ipopt", {"print_time": False}, {"print_level": 0, "sb": "yes"})
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError:
        sol = opti.debug
        ok = False

    Th = np.asarray(sol.value(Thouse)).ravel()
    return dict(
        ok=ok,
        status=opti.stats()["return_status"],
        iters=opti.stats()["iter_count"],
        cost=float(sol.value(energy_cost)),
        violation=float(np.sum(np.maximum(0, 20 - Th[1:]) + np.maximum(0, Th[1:] - 23))),  # K·h
        Thouse=Th,
        Tfloor=np.asarray(sol.value(Tfloor)).ravel(),
        Qth=np.asarray(sol.value(Qth)).ravel(),
        P=np.asarray(sol.value(P)).ravel(),
    )


# ============ A) sensitivity to initialization ============
Thouse0, Tfloor0 = 20, 22  # same initial state as ex4.py

Qth_sim = np.full(nsteps, 3600.0)
Th_sim, Tf_sim = simulate(Thouse0, Tfloor0, Qth_sim)

rng = np.random.default_rng(0)
guesses = {
    "nessuna (zeri)": None,
    "come ex4 (20 / 25 / 3000)": dict(Thouse=20, Tfloor=25, Qth=3000),
    "regime (21 / 28 / 3600)": dict(Thouse=21, Tfloor=28, Qth=3600),
    "freddo (15 / 15 / 0)": dict(Thouse=15, Tfloor=15, Qth=0),
    "caldo (25 / 40 / 10000)": dict(Thouse=25, Tfloor=40, Qth=10000),
    "simulazione (Qth=3600)": dict(Thouse=Th_sim, Tfloor=Tf_sim, Qth=Qth_sim),
}
for i in range(3):
    guesses[f"casuale {i}"] = dict(
        Thouse=rng.uniform(15, 25, nsteps + 1),
        Tfloor=rng.uniform(18, 40, nsteps + 1),
        Qth=rng.uniform(0, 8000, nsteps),
    )

print("A) Sensitivity to initialization (Thouse0 = 20, Tfloor0 = 22)")
print(f"{'guess':<28} {'status':<28} {'iter':>5} {'cost €':>8} {'viol K·h':>9}")
results_A = {}
for name, guess in guesses.items():
    r = solve_ocp(Thouse0, Tfloor0, guess)
    results_A[name] = r
    if r["ok"]:
        print(f"{name:<28} {r['status']:<28} {r['iters']:>5} {r['cost']:8.4f} {r['violation']:9.3f}")
    else:
        print(f"{name:<28} {r['status']:<28} {r['iters']:>5} {'-':>8} {'-':>9}")

fig = plt.figure(figsize=(11, 13))
ax0 = fig.add_subplot(4, 1, 1)
axes = [ax0, fig.add_subplot(4, 1, 2, sharex=ax0), fig.add_subplot(4, 1, 3, sharex=ax0), fig.add_subplot(4, 1, 4)]
fig.suptitle("Es. 4 – sensibilità all'inizializzazione (stesso problema, set_initial diversi)")
edges = np.arange(nsteps + 1)
for i, (name, r) in enumerate(results_A.items()):
    if not r["ok"]:
        continue  # failed solves have no meaningful trajectory
    style = dict(color=f"C{i}", linestyle=["-", "--", ":", "-."][i % 4], alpha=0.8)
    axes[0].plot(edges, r["Thouse"], marker=".", label=name, **style)
    axes[1].plot(edges, r["Tfloor"], marker=".", **style)
    axes[2].step(edges, np.append(r["P"], r["P"][-1]), where="post", **style)
for y in (20, 23):
    axes[0].axhline(y, color="gray", linestyle=":")
axes[0].set_ylabel("T aria [°C]")
axes[0].legend(loc="lower right", fontsize=8)
axes[1].set_ylabel("T pavimento [°C]")
axes[2].set_ylabel("P_el [W]")
ax_price = axes[2].twinx()
ax_price.step(edges, np.append(prices, prices[-1]), where="post", color="tab:orange", alpha=0.4)
ax_price.set_ylabel("Prezzo [€/kWh]")

names = list(results_A)
colors = ["tab:blue" if results_A[n]["ok"] else "tab:red" for n in names]
axes[3].bar(range(len(names)), [results_A[n]["iters"] for n in names], color=colors)
axes[3].set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=8)
axes[3].set_ylabel("Iterazioni IPOPT\n(rosso = fallito)")
for ax in axes[:3]:
    ax.grid(alpha=0.3)
axes[3].grid(alpha=0.3, axis="y")
plt.tight_layout()

# ============ B) sensitivity to the initial state ============
Tfloor0_list = [20, 22, 25, 28, 31]

print("\nB) Sensitivity to the initial floor temperature (Thouse0 = 20)")
print(f"{'Tfloor0':>8} {'status':<28} {'cost €':>8} {'viol K·h':>9} {'T aria(1)':>10}")
results_B = {}
for Tf0 in Tfloor0_list:
    r = solve_ocp(20, Tf0, dict(Thouse=21, Tfloor=28, Qth=3600))
    results_B[Tf0] = r
    print(f"{Tf0:>8} {r['status']:<28} {r['cost']:8.3f} {r['violation']:9.3f} {r['Thouse'][1]:10.2f}")

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle("Es. 4 – sensibilità allo stato iniziale del pavimento (T aria(0) = 20 °C)")
for i, (Tf0, r) in enumerate(results_B.items()):
    label = f"Tfloor(0) = {Tf0} °C  ({r['cost']:.2f} €, {r['violation']:.2f} K·h)"
    axes[0].plot(edges, r["Thouse"], marker=".", color=f"C{i}", label=label)
    axes[1].plot(edges, r["Tfloor"], marker=".", color=f"C{i}")
    axes[2].step(edges, np.append(r["P"], r["P"][-1]), where="post", color=f"C{i}")
for y in (20, 23):
    axes[0].axhline(y, color="gray", linestyle=":")
axes[0].set_ylabel("T aria [°C]")
axes[0].legend(loc="lower right", fontsize=8)
axes[1].set_ylabel("T pavimento [°C]")
axes[2].set_ylabel("P_el [W]")
axes[2].set_xlabel("Time [h]")
ax_price = axes[2].twinx()
ax_price.step(edges, np.append(prices, prices[-1]), where="post", color="tab:orange", alpha=0.4)
ax_price.set_ylabel("Prezzo [€/kWh]")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()

plt.show()
