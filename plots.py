"""Shared metrics printout and 3-panel plot for the heat-pump exercises."""

import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX = 20, 23  # comfort limits, °C


def print_metrics(T, P, prices, dt=3600):
    """T: indoor temperature (n+1 values, °C), P: electric power (n values, W), prices: €/kWh (at least n values)."""
    T = np.asarray(T, dtype=np.float64).ravel()
    P = np.asarray(P, dtype=np.float64).ravel()
    n = len(P)
    prices = np.asarray(prices, dtype=np.float64).ravel()[:n]

    cost = np.sum(prices * P / 1000 * dt / 3600)  # €
    energy = np.sum(P / 1000 * dt / 3600)  # kWh electric
    T_after = T[1:]  # temperature at the end of each hour (T[0] is the initial condition)
    violation = np.sum(np.maximum(0, T_MIN - T_after) + np.maximum(0, T_after - T_MAX)) * dt / 3600  # K·h

    print(f"Cost:              {cost:.2f} € over {n * dt / 3600:.0f} h")
    print(f"Electric energy:   {energy:.1f} kWh")
    print(f"Comfort violation: {violation:.3f} K·h")
    print(f"T min / max:       {T_after.min():.2f} / {T_after.max():.2f} °C")


def plot_results(T, P, prices, Text, Text_real=None, setpoint=None, title=None):
    """3 panels: indoor temperature, power + price, outdoor temperature. Same shapes as print_metrics."""
    T = np.asarray(T, dtype=np.float64).ravel()
    P = np.asarray(P, dtype=np.float64).ravel()
    n = len(P)
    prices = np.asarray(prices, dtype=np.float64).ravel()[:n]
    Text = np.asarray(Text, dtype=np.float64).ravel()[:n]
    hours = np.arange(n)
    hours_edges = np.arange(n + 1)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    if title:
        fig.suptitle(title)

    # 1) indoor temperature
    axes[0].plot(hours_edges, T, color="tab:red", marker=".", label="T indoor")
    axes[0].axhline(T_MIN, color="gray", linestyle="--", label="comfort limits")
    axes[0].axhline(T_MAX, color="gray", linestyle="--")
    if setpoint is not None:
        axes[0].axhline(setpoint, color="black", linestyle=":", label="setpoint")
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].legend(loc="upper right")

    # 2) electric power and price (piecewise constant over each hour)
    axes[1].step(hours_edges, np.append(P, P[-1]), where="post", color="tab:blue", label="P_el")
    axes[1].set_ylabel("Power [W]")
    ax_price = axes[1].twinx()
    ax_price.step(hours_edges, np.append(prices, prices[-1]), where="post", color="tab:orange", label="price")
    ax_price.set_ylabel("Price [€/kWh]")
    lines = axes[1].get_lines() + ax_price.get_lines()
    axes[1].legend(lines, [l.get_label() for l in lines], loc="upper right")

    # 3) outdoor temperature (forecast vs real when they differ)
    if Text_real is None:
        axes[2].plot(hours, Text, color="tab:green", label="T_a")
    else:
        Text_real = np.asarray(Text_real, dtype=np.float64).ravel()[:n]
        axes[2].plot(hours, Text, color="tab:green", label="T_a forecast")
        axes[2].plot(hours, Text_real, color="tab:green", linestyle=":", marker=".", label="T_a real")
    axes[2].set_ylabel("Outdoor temp. [°C]")
    axes[2].set_xlabel("Time [h]")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_results_2states(runs, prices, Text, title=None):
    """runs: {label: dict(Thouse, Tfloor, P, COP, s_low)}, one line per run. 5 panels: air, floor, power + price, COP, slack."""
    first = next(iter(runs.values()))
    n = len(np.asarray(first["P"]).ravel())
    prices = np.asarray(prices, dtype=np.float64).ravel()[:n]
    Text = np.asarray(Text, dtype=np.float64).ravel()[:n]
    hours = np.arange(n)
    hours_edges = np.arange(n + 1)
    width = 0.8 / len(runs)

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)
    if title:
        fig.suptitle(title)

    for i, (label, r) in enumerate(runs.items()):
        Thouse = np.asarray(r["Thouse"], dtype=np.float64).ravel()
        Tfloor = np.asarray(r["Tfloor"], dtype=np.float64).ravel()
        P = np.asarray(r["P"], dtype=np.float64).ravel()
        COP = np.asarray(r["COP"], dtype=np.float64).ravel()
        s_low = np.asarray(r["s_low"], dtype=np.float64).ravel()
        color = f"C{i}"
        linestyle = "-" if i == 0 else "--"

        # 1) air temperature
        axes[0].plot(hours_edges, Thouse, color=color, linestyle=linestyle, marker=".", label=f"T aria {label}")
        # 2) floor temperature
        axes[1].plot(hours_edges, Tfloor, color=color, linestyle=linestyle, marker=".", label=f"T pavimento {label}")
        # 3) electric power (piecewise constant over each hour)
        axes[2].step(hours_edges, np.append(P, P[-1]), where="post", color=color, linestyle=linestyle, label=f"P_el {label}")
        # 4) COP during each hour
        axes[3].step(hours_edges, np.append(COP, COP[-1]), where="post", color=color, linestyle=linestyle, label=f"COP {label}")
        # 5) lower-comfort slack, bars side by side (s_low[k] relaxes Thouse[k+1])
        axes[4].bar(hours + 1 + (i - (len(runs) - 1) / 2) * width, s_low, width=width, color=color, label=f"s_low {label}")

    axes[0].axhline(T_MIN, color="gray", linestyle=":", label="comfort limits")
    axes[0].axhline(T_MAX, color="gray", linestyle=":")
    axes[0].set_ylabel("T aria [°C]")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_ylabel("T pavimento [°C]")
    ax_text = axes[1].twinx()
    ax_text.plot(hours, Text, color="tab:green", alpha=0.6, label="T esterna")
    ax_text.set_ylabel("T esterna [°C]")
    lines = axes[1].get_lines() + ax_text.get_lines()
    axes[1].legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)

    axes[2].set_ylabel("P_el [W]")
    ax_price = axes[2].twinx()
    ax_price.step(hours_edges, np.append(prices, prices[-1]), where="post", color="tab:orange", alpha=0.6, label="prezzo")
    ax_price.set_ylabel("Prezzo [€/kWh]")
    lines = axes[2].get_lines() + ax_price.get_lines()
    axes[2].legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)

    axes[3].set_ylabel("COP [-]")
    axes[3].legend(loc="upper right", fontsize=8)

    axes[4].set_ylabel("Slack inferiore [K]")
    axes[4].set_xlabel("Time [h]")
    axes[4].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
