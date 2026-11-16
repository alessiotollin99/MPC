import casadi as cd
import numpy as np

"""
Stato iniziale T(0) = 20
Tbounded = 20 <= T <= 23
Pelbounded = 0 <= P <= 3000
"""
# input data selection
C = 1 * 10**7  # J/K
UA = 200  # W/K
COP = 3  # (costante)
P_elmin = 0  # W
Pel_max = 1500  # W

dt = 3600  # s
n_steps = 24  # hours
Tin = 20  # °C

time_series = np.arange(0, dt * n_steps, dt)  # temporal series

# external temperature
f_Text = lambda t: (
    3 + 4 * np.sin(2 * np.pi * (t - 9) / 24)
)  # external temperature function
Text = f_Text(time_series / 3600)
Text = np.full(n_steps, -15)

# prices definition
hours = np.arange(0, n_steps, dtype=np.float64) % 24  # number of hour per day
prices = hours.copy()
mask_6 = hours < 6
mask_17_21 = (hours >= 17) & (hours < 21)
prices[mask_6] = 0.12
prices[mask_17_21] = 0.40
prices[~mask_6 & ~mask_17_21] = 0.20
prices = prices[:, None]

rho = 0.5 # € / K price for violating slack

# optimization variables creation

opti = cd.Opti()

P = opti.variable(n_steps)
T = opti.variable(n_steps + 1)
s = opti.variable(n_steps) # slack variable

obj = cd.sum1(P * prices / 1000 * dt / 3600) + rho*cd.sum1(s**2)  # W -> kW, dt s -> h: cost in €

con_temperature = T[1:] == T[:-1] + dt / C * (
    UA * (Text[:] - T[:-1]) + COP * P
)  # temperature constraint to respect

opti.minimize(obj)

opti.subject_to(opti.bounded(P_elmin, P, Pel_max))  # power limited between 0 and 3000 W
opti.subject_to(s >= 0) # slack constraint for unfeasability
opti.subject_to(T[1:] >= 20 - s) # slack constraint for unf.
opti.subject_to(T[1:] <= 23)  # comfort temperature below 23 °C
opti.subject_to(T[0] == Tin)  # initial temperature at 20 °C

opti.subject_to(con_temperature)

# dovremmo impostare anche la soluzione iniziale ma al momento non facciamo perchè il problema è convesso
opti.solver("ipopt")
sol = opti.solve()

from plots import print_metrics, plot_results

print_metrics(sol.value(T), sol.value(P), prices)
plot_results(sol.value(T), sol.value(P), prices, Text, title="Open-loop OCP")
