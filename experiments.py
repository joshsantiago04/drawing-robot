import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from tremor import generate_path, add_tremor
from kalman import filter_path

np.random.seed(42)
true_path = generate_path(shape="circle", n_points=300)


# --- Experiment 0: shape comparison ---
shape_configs = [
    {"shape": "circle",  "label": "Circle"},
    {"shape": "figure8", "label": "Figure 8"},
    {"shape": "spiral",  "label": "Spiral"},
]

fig0, axes0 = plt.subplots(3, 3, figsize=(14, 13))
fig0.suptitle("Kalman Filter Across Different Drawing Shapes", fontsize=14, y=1.01)

for row, cfg in enumerate(shape_configs):
    path     = generate_path(shape=cfg["shape"], n_points=300)
    noisy    = add_tremor(path, frequency=60.0, amplitude=0.055)
    filtered = filter_path(noisy, process_noise=1e-3, measurement_noise=0.5)

    ne = np.linalg.norm(noisy    - path, axis=1).mean()
    fe = np.linalg.norm(filtered - path, axis=1).mean()
    reduction = (1 - fe / ne) * 100

    titles = [
        cfg["label"],
        f"With Tremor\nMean err: {ne:.4f}",
        f"Kalman Filtered\nMean err: {fe:.4f}  ({reduction:.1f}% reduction)",
    ]
    paths  = [path, noisy, filtered]
    colors = ["green", "red", "blue"]

    for col, (p, title, color) in enumerate(zip(paths, titles, colors)):
        ax = axes0[row][col]
        ax.plot(p[:, 0], p[:, 1], color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.autoscale(enable=True, tight=True)

plt.tight_layout()
fig0.savefig("results/results_shapes.png", dpi=150, bbox_inches="tight")
print("Saved results_shapes.png")


# --- Experiment 1: vary tremor frequency ---
freq_configs = [
    {"frequency": 10.0,  "amplitude": 0.055, "label": "10 Hz (mild)",     "pn": 1e-4, "mn": 1.0},
    {"frequency": 25.0,  "amplitude": 0.055, "label": "25 Hz (moderate)", "pn": 1e-3, "mn": 0.5},
    {"frequency": 60.0,  "amplitude": 0.055, "label": "60 Hz (severe)",   "pn": 1e-3, "mn": 0.5},
]

fig, axes = plt.subplots(3, 3, figsize=(14, 13))
fig.suptitle("Effect of Tremor Frequency on Kalman Filtering", fontsize=14, y=1.01)

for row, cfg in enumerate(freq_configs):
    noisy    = add_tremor(true_path, frequency=cfg["frequency"], amplitude=cfg["amplitude"])
    filtered = filter_path(noisy, process_noise=cfg["pn"], measurement_noise=cfg["mn"])

    ne = np.linalg.norm(noisy    - true_path, axis=1).mean()
    fe = np.linalg.norm(filtered - true_path, axis=1).mean()
    reduction = (1 - fe / ne) * 100

    titles  = ["Ground Truth", f"Tremor ({cfg['label']})\nMean err: {ne:.4f}", f"Kalman Filtered\nMean err: {fe:.4f}  ({reduction:.1f}% reduction)"]
    paths   = [true_path, noisy, filtered]
    colors  = ["green", "red", "blue"]

    for col, (path, title, color) in enumerate(zip(paths, titles, colors)):
        ax = axes[row][col]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("results/results_frequency.png", dpi=150, bbox_inches="tight")
print("Saved results_frequency.png")


# --- Experiment 2: vary tremor amplitude ---
amp_configs = [
    {"frequency": 60.0, "amplitude": 0.03,  "label": "Low (0.03)"},
    {"frequency": 60.0, "amplitude": 0.055, "label": "Medium (0.055)"},
    {"frequency": 60.0, "amplitude": 0.12,  "label": "High (0.12)"},
]

fig2, axes2 = plt.subplots(3, 3, figsize=(14, 13))
fig2.suptitle("Effect of Tremor Amplitude on Kalman Filtering", fontsize=14, y=1.01)

for row, cfg in enumerate(amp_configs):
    noisy    = add_tremor(true_path, frequency=cfg["frequency"], amplitude=cfg["amplitude"])
    filtered = filter_path(noisy, process_noise=1e-3, measurement_noise=0.5)

    ne = np.linalg.norm(noisy    - true_path, axis=1).mean()
    fe = np.linalg.norm(filtered - true_path, axis=1).mean()
    reduction = (1 - fe / ne) * 100

    titles = ["Ground Truth", f"Tremor (amp={cfg['label']})\nMean err: {ne:.4f}", f"Kalman Filtered\nMean err: {fe:.4f}  ({reduction:.1f}% reduction)"]
    paths  = [true_path, noisy, filtered]
    colors = ["green", "red", "blue"]

    for col, (path, title, color) in enumerate(zip(paths, titles, colors)):
        ax = axes2[row][col]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.8, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig("results/results_amplitude.png", dpi=150, bbox_inches="tight")
print("Saved results_amplitude.png")


# --- Experiment 3: summary error bar chart ---
configs = [
    {"frequency": 10.0,  "amplitude": 0.055, "pn": 1e-4, "mn": 1.0},
    {"frequency": 25.0,  "amplitude": 0.055, "pn": 1e-3, "mn": 0.5},
    {"frequency": 60.0,  "amplitude": 0.055, "pn": 1e-3, "mn": 0.5},
    {"frequency": 60.0,  "amplitude": 0.03,  "pn": 1e-3, "mn": 0.5},
    {"frequency": 60.0,  "amplitude": 0.12,  "pn": 1e-3, "mn": 0.5},
]
labels        = ["10 Hz", "25 Hz", "60 Hz", "60 Hz\nlow amp", "60 Hz\nhigh amp"]
noisy_errors  = []
filtered_errors = []

for cfg in configs:
    noisy    = add_tremor(true_path, frequency=cfg["frequency"], amplitude=cfg["amplitude"])
    filtered = filter_path(noisy, process_noise=cfg["pn"], measurement_noise=cfg["mn"])
    noisy_errors.append(   np.linalg.norm(noisy    - true_path, axis=1).mean())
    filtered_errors.append(np.linalg.norm(filtered - true_path, axis=1).mean())

x = np.arange(len(labels))
width = 0.35

fig3, ax3 = plt.subplots(figsize=(10, 5))
bars1 = ax3.bar(x - width/2, noisy_errors,    width, label="With Tremor",     color="salmon")
bars2 = ax3.bar(x + width/2, filtered_errors, width, label="Kalman Filtered", color="steelblue")

for bar, val in zip(bars1, noisy_errors):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars2, filtered_errors):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax3.set_xlabel("Configuration")
ax3.set_ylabel("Mean Position Error")
ax3.set_title("Mean Position Error: Tremor vs. Kalman Filtered")
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend()
ax3.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig3.savefig("results/results_summary.png", dpi=150)
print("Saved results_summary.png")

# Print table
print("\n--- Summary Table ---")
print(f"{'Config':<20} {'Noisy Error':>12} {'Filtered Error':>15} {'Reduction':>10}")
print("-" * 60)
for label, ne, fe in zip(labels, noisy_errors, filtered_errors):
    print(f"{label:<20} {ne:>12.4f} {fe:>15.4f} {(1-fe/ne)*100:>9.1f}%")


# --- Experiment 4: FFT frequency analysis ---
# Shows the tremor frequency spike in the raw signal and its suppression after filtering.
SAMPLE_RATE = 300

noisy_fft    = add_tremor(true_path, frequency=60.0, amplitude=0.055)
filtered_fft = filter_path(noisy_fft, process_noise=1e-3, measurement_noise=0.5)

# Use the x-axis signal for FFT
signal_noisy    = noisy_fft[:, 0] - true_path[:, 0]   # just the noise component
signal_filtered = filtered_fft[:, 0] - true_path[:, 0]

n = len(signal_noisy)
freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
fft_noisy    = np.abs(np.fft.rfft(signal_noisy))
fft_filtered = np.abs(np.fft.rfft(signal_filtered))

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(freqs, fft_noisy,    label="With Tremor",     color="red",      alpha=0.8)
ax4.plot(freqs, fft_filtered, label="Kalman Filtered", color="steelblue", alpha=0.9)
ax4.axvline(60.0, color="gray", linestyle="--", linewidth=1, label="Tremor frequency (60 Hz)")
ax4.set_xlim(0, 150)
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Amplitude")
ax4.set_title("Frequency Spectrum: Tremor vs. Kalman Filtered (FFT)")
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig("results/results_fft.png", dpi=150)
print("Saved results_fft.png")


# --- Experiment 5: PID vs instant IK tracking ---
from robot import PlanarArm

arm = PlanarArm()
_, tip_instant = arm.follow_path(filtered_fft)
_, tip_pid     = arm.follow_path_pid(filtered_fft, steps_per_waypoint=20)

# Downsample instant to match PID output length for fair comparison
pid_len = len(tip_pid)
instant_idx = np.linspace(0, len(tip_instant) - 1, pid_len).astype(int)
tip_instant_ds = tip_instant[instant_idx]

# Ground truth scaled the same way robot.py scales targets
true_scaled = np.column_stack((true_path[:, 0] * 0.5 + 0.9, true_path[:, 1] * 0.5))
true_idx = np.linspace(0, len(true_scaled) - 1, pid_len).astype(int)
true_ds = true_scaled[true_idx]

err_instant = np.linalg.norm(tip_instant_ds - true_ds, axis=1).mean()
err_pid     = np.linalg.norm(tip_pid        - true_ds, axis=1).mean()

fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
fig5.suptitle("Robot Arm Tracking: Instant IK vs. PID Control", fontsize=13)

for ax, tips, label, color, err in [
    (axes5[0], tip_instant_ds, "Instant IK",   "darkorange", err_instant),
    (axes5[1], tip_pid,        "PID Control",  "steelblue",  err_pid),
]:
    ax.plot(true_ds[:, 0],  true_ds[:, 1],  color="green", linewidth=1,
            linestyle="--", alpha=0.5, label="Target path")
    ax.plot(tips[:, 0], tips[:, 1], color=color, linewidth=0.8, alpha=0.85, label=label)
    ax.set_title(f"{label}\nMean tip error: {err:.4f}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 2.0)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig5.savefig("results/results_pid.png", dpi=150)
print("Saved results_pid.png")
print(f"\nInstant IK tip error: {err_instant:.4f}")
print(f"PID tip error:        {err_pid:.4f}")
