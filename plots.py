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
