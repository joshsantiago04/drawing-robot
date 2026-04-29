# main.py
# Entry point for the drawing robot demo.
#
# Runs the full pipeline for each shape:
#   1. Generate a clean path
#   2. Add simulated tremor
#   3. Apply Kalman filter
#   4. Save comparison and error plots to results/
#   5. Open interactive robot arm animations
#
# Run: venv/bin/python main.py

import numpy as np
import matplotlib.pyplot as plt

from tremor import generate_path, add_tremor
from kalman import filter_path
from robot import PlanarArm
from visualize import plot_comparison, plot_error, animate_arm

np.random.seed(42)  # fixed seed for reproducible results across runs

SHAPES      = ["circle", "figure8", "spiral"]
TREMOR_FREQ = 60.0   # Hz — oscillation frequency of the simulated tremor
TREMOR_AMP  = 0.055  # amplitude of tremor relative to drawing size
arm         = PlanarArm(l1=1.0, l2=0.8)

# Keep animation references alive — Python will close windows if these get GC'd
animations = []

for shape in SHAPES:
    # --- Pipeline ---
    true_path     = generate_path(shape=shape, n_points=300)
    noisy_path    = add_tremor(true_path, frequency=TREMOR_FREQ, amplitude=TREMOR_AMP)
    filtered_path = filter_path(noisy_path, process_noise=1e-3, measurement_noise=0.5)

    # --- Evaluate ---
    noisy_err    = np.linalg.norm(noisy_path - true_path, axis=1).mean()
    filtered_err = np.linalg.norm(filtered_path - true_path, axis=1).mean()
    improvement  = (1 - filtered_err / noisy_err) * 100

    print(f"[{shape}]  noisy={noisy_err:.4f}  filtered={filtered_err:.4f}  reduction={improvement:.1f}%")

    # --- Save plots ---
    fig1 = plot_comparison(true_path, noisy_path, filtered_path,
                           title=f"Tremor Suppression — {shape.capitalize()}")
    fig1.savefig(f"results/comparison_{shape}.png", dpi=150)

    fig2 = plot_error(true_path, noisy_path, filtered_path)
    fig2.savefig(f"results/error_plot_{shape}.png", dpi=150)

    # --- Animate ---
    fig3, ani = animate_arm(arm, filtered_path, true_path,
                            title=f"Robot Arm — {shape.capitalize()}")
    animations.append((fig3, ani))

print("\nAll animations open — close windows to exit.")
plt.show()
