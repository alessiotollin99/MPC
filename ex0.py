import numpy as np
import casadi as cd

C = 1*10**7 #J/K
UA = 200 #W/K
COP	= 3 #(costante)
P_elmin = 0 #W
Pel_max = 3000 #W
dt = 3600 #s
n_steps = 48 #hours
Tin = 20 #°C

f_Text = lambda t: 3+4*np.sin(2*np.pi*(t-9)/24) # external temperature function

time_series = np.arange(0, dt*n_steps, dt) # temporal series
Text = f_Text(time_series / 3600)
P = {P_elmin: [], Pel_max: []}

def myRC_onestep(UA, Pel, t, Text):
        
    T = Tvec[t] + dt/C * (UA * (Text[t]-Tvec[t]) + COP * Pel)
    return T

for p in P.keys():
    Tvec = np.zeros(n_steps+1)
    Tvec[0] = Tin
    for t in range(n_steps):
        Tvec[t+1] = myRC_onestep(UA, p, t, Text)
    P[p].append(Text)
    P[p].append(Tvec)

#======================================================================
# Figures
#======================================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize = (11, 5), sharey=True)
for p, ax in zip(P.keys(), axes):
    ax.plot(np.arange(n_steps + 1), P[p][1], color="tab:red", marker=".", label="T indoor")
    ax.plot(np.arange(n_steps), P[p][0], color="tab:green", label="T_a")
    ax.axhline(20, color="gray", linestyle="--", label="comfort limits")
    ax.axhline(23, color="gray", linestyle="--")
    ax.set_title(f"Constant P_el = {p} W")
    ax.set_xlabel("Time [h]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
axes[0].set_ylabel("Temperature [°C]")

fig.tight_layout()
plt.show()

#======================================================================
# PI control + windup
#======================================================================

Tvec = np.zeros(n_steps+1)
Tvec[0] = Tin
errors = np.zeros(n_steps)
Pel = np.zeros(n_steps)
Kp = 1000 #W/K
Ki = 0.3
Itry = 0
I = 0

for t in range(n_steps):
    errors[t] = 21 - Tvec[t]
    # Itry = Itry + errors[t]*dt
    Itry = I + errors[t]*dt
    Ptry = Kp*errors[t]+Ki*Itry
    # Pel[t] = np.clip(Ptry, P_elmin, Pel_max)
    if (Ptry >= P_elmin) and (Ptry <= Pel_max):
        I = Itry
        Pel[t] = Ptry
    else:
        Pel[t]=np.clip(Ptry, P_elmin, Pel_max)

    Tvec[t+1] = myRC_onestep(UA, Pel[t], t, Text)

#======================================================================
# Electricity prices
#======================================================================

hours = np.arange(0, n_steps, dtype = np.float64) % 24 # number of hour per day
prices = hours.copy()
mask_6 = hours < 6
mask_17_21 = (hours >= 17) & (hours < 21)
prices[mask_6] = 0.12
prices[mask_17_21] = 0.40
prices[~mask_6 & ~mask_17_21] = 0.20

from plots import print_metrics, plot_results

print_metrics(Tvec, Pel, prices)
plot_results(Tvec, Pel, prices, Text, setpoint=21, title="PI control + anti-windup")

# Ore di discomfort: 0, Euro spesi: 12.12 -> esito del controllo PI + wind-up