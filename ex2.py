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
Pel_max = 3000  # W

dt = 3600  # s
n_steps = int(input("How many steps do you want to simulate?\n"))  # 72 hours
N_prevision = 24 #hours to simulate after n_steps
Tin = 20  # °C

# for randomic component
rng = np.random.default_rng(seed=0)
casualties = rng.normal(0, 1, N_prevision+n_steps)

time_series = np.arange(0, dt * (n_steps + N_prevision), dt)  # temporal series

# external temperature
f_Text = lambda t: (
    3 + 4 * np.sin(2 * np.pi * (t - 9) / 24)
)  # external temperature function
Text = f_Text(time_series / 3600)
Text_real = Text + casualties

# init
Tvec = np.zeros(n_steps+1, dtype = np.float32)
Pvec = np.zeros(n_steps, dtype=np.float32)

Tvec[0] = Tin

# prices definition
hours = np.arange(0, n_steps+N_prevision, dtype=np.float64) % 24  # number of hour per day
prices = hours.copy()
mask_6 = hours < 6
mask_17_21 = (hours >= 17) & (hours < 21)
prices[mask_6] = 0.12
prices[mask_17_21] = 0.40
prices[~mask_6 & ~mask_17_21] = 0.20
prices = prices[:, None]

# optimization variables creation

opti = cd.Opti()

Tin_loc = opti.parameter()
Text_prev = opti.parameter(N_prevision)
prices_prev = opti.parameter(N_prevision)

P = opti.variable(N_prevision)
T = opti.variable(N_prevision + 1)

opti.subject_to(opti.bounded(P_elmin, P, Pel_max))  # power limited between 0 and 3000 W
opti.subject_to(opti.bounded(20, T[1:], 23))  # comfort temperature between 20 and 23 °C

obj = cd.sum1(P * prices_prev / 1000 * dt / 3600)  # W -> kW, dt s -> h: cost in €

con_temperature = T[1:] == T[:-1] + dt / C * (
    UA * (Text_prev - T[:-1]) + COP * P
)  # temperature constraint to respect

opti.minimize(obj)

opti.subject_to(T[0] == Tin_loc)  # initial temperature at 20 °C

opti.subject_to(con_temperature)

# dovremmo impostare anche la soluzione iniziale ma al momento non facciamo perchè il problema è convesso
opti.solver("ipopt")

def myRC_onestep(UA, Pel, t, Text):
        
    T = Tvec[t] + dt/C * (UA * (Text[t]-Tvec[t]) + COP * Pel)
    return T

def my_optimization(Tvec, Text,prices,t, N_prevision):
    opti.set_value(Tin_loc, Tvec[t])
    opti.set_value(Text_prev, Text[t:t+N_prevision])
    opti.set_value(prices_prev, prices[t:t+N_prevision])
    
    sol = opti.solve()

    return sol.value(P)

for t in range(n_steps):
    P_loc = my_optimization(Tvec, Text, prices, t, N_prevision)[0]
    Pvec[t] = P_loc
    Tvec[t+1] = myRC_onestep(220, P_loc, t, Text_real)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

ax_left = axes[0]
ax_right = ax_left.twinx()

ax_left.plot(
     time_series[:-N_prevision] / dt, Tvec[:-1], color="tab:red", label="Air temperature"
)
ax_right.plot(time_series[:-N_prevision] / dt, Pvec, color="tab:blue", label="Power")

ax_left.set_ylabel("Temperature [°C]")
ax_right.set_ylabel("Power [W]")
ax_left.set_xlabel("Time [h]")

lines = ax_left.get_lines() + ax_right.get_lines()
ax_left.legend(lines, [l.get_label() for l in lines], loc="upper left")

# axes[1].plot(
#     time_series / dt,
#     sol.value(P) * prices.reshape(-1) / 1000 * dt / 3600,
#     color="black",
#     linestyle="-.",
# )
# axes[1].set_xlabel("Time [h]")
# axes[1].set_ylabel("Hourly cost [€]")

# cost = sol.value(obj)

# axes[1].annotate(
#     f"Minimum cost: {cost:.2f} €",
#     xy=(0.02, 0.95),
#     xycoords="axes fraction",
#     va="top",
#     bbox=dict(boxstyle="round", fc="white", ec="black"),
# )

plt.tight_layout()
plt.show()
