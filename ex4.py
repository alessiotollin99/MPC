import casadi as cd
import numpy as np

# -------------house ------------------------
UA_home = 200  # W/K air-outdoor
C_home = 2 * 10**6  # J/K

# -------------floor-------------------------
UA_floor = 500  # W/k air-floor
C_floor = 1 * 10**7
# -------------temporal part-----------------
dt = 3600  # s
nsteps = 24  # hours
# nsteps
time_series = np.arange(0, dt * nsteps, dt)

# -------------external temperature----------
f_Text = lambda t: (
    3 + 4 * np.sin(2 * np.pi * (t - 9) / 24)
)  # external temperature function
Text = f_Text(time_series / 3600)

# -------------prices definition-------------
hours = np.arange(0, nsteps, dtype=np.float64) % 24  # number of hour per day
prices = hours.copy()
mask_6 = hours < 6
mask_17_21 = (hours >= 17) & (hours < 21)
prices[mask_6] = 0.12
prices[mask_17_21] = 0.40
prices[~mask_6 & ~mask_17_21] = 0.20
prices = prices[:, None]

# -------------optimization problem----------
opti = cd.Opti()

Thouse = opti.variable(nsteps + 1)
Tfloor = opti.variable(nsteps + 1)
Qth = opti.variable(nsteps)
s_low = opti.variable(nsteps)  # slack variable lower end
s_upp = opti.variable(nsteps) # slack variable upper end
rho = 0.2  # €/K violation cost

Tsupply = Tfloor[:-1] + 5
COP = 0.45 * (Tsupply + 273.15) / (Tsupply-Text)
P = Qth / COP
obj = cd.sum1(P * prices / 1000 * dt / 3600) + rho * cd.sum1(s_low) + rho * cd.sum1(s_upp)
opti.minimize(obj)

con_Thouse = Thouse[1:] == Thouse[:-1] + dt / C_home * UA_home * (
    Text - Thouse[:-1]
) + dt / C_home * UA_floor * (Tfloor[:-1] - Thouse[:-1])
con_Tfloor = Tfloor[1:] == Tfloor[
    :-1
] + dt / C_floor * Qth - dt / C_floor * UA_floor * (Tfloor[:-1] - Thouse[:-1])

opti.subject_to(opti.bounded(0, P, 3000))
opti.subject_to(s_low >= 0)
opti.subject_to(s_upp>=0)
opti.subject_to(Thouse[1:] >= 20 - s_low)
opti.subject_to(Thouse[1:] <= 23 + s_upp)
opti.subject_to(Qth >= 0)
opti.subject_to(Tfloor[0] == 22)
opti.subject_to(Thouse[0] == 20)

opti.subject_to(con_Tfloor)
opti.subject_to(con_Thouse)

opti.set_initial(Thouse, 20)
opti.set_initial(Tfloor, 25)
opti.set_initial(Qth, 3000)
opti.set_initial(s_low, 0)
opti.set_initial(s_upp, 0)

opti.solver("ipopt")
sol = opti.solve()

# -------------results: linear vs quadratic penalty----------
from plots import print_metrics, plot_results_2states

def get_results(sol):
    return dict(
        Thouse=sol.value(Thouse),
        Tfloor=sol.value(Tfloor),
        P=sol.value(P),
        COP=sol.value(COP),
        s_low=sol.value(s_low),
    )

res_lin = get_results(sol)

# same problem, quadratic penalty instead of linear (constraints unchanged)
obj_quad = cd.sum1(P * prices / 1000 * dt / 3600) + rho * cd.sum1(s_low**2) + rho * cd.sum1(s_upp**2)
opti.minimize(obj_quad)
sol_quad = opti.solve()
res_quad = get_results(sol_quad)

print("\n--- linear penalty ---")
print_metrics(res_lin["Thouse"], res_lin["P"], prices)
print("--- quadratic penalty ---")
print_metrics(res_quad["Thouse"], res_quad["P"], prices)

plot_results_2states(
    {"lineare": res_lin, "quadratica": res_quad},
    prices,
    Text,
    title=f"Es. 4 open loop – penalità lineare vs quadratica, rho = {rho}",
)
