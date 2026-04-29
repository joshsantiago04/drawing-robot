# tremor.py
# Generates clean drawing paths and simulates involuntary hand tremor.
# The tremor model is based on Parkinsonian tremor: a sinusoidal oscillation
# in the 4-12 Hz range plus small random gaussian noise on top.

import numpy as np


def generate_path(shape="circle", n_points=300):
    """Generate a clean intended drawing path (no tremor)."""
    # Parameter t sweeps 0 → 2π, giving one full revolution of the shape
    t = np.linspace(0, 2 * np.pi, n_points)

    if shape == "circle":
        x = np.cos(t)
        y = np.sin(t)
    elif shape == "figure8":
        # Lissajous curve with 2:1 frequency ratio
        x = np.sin(t)
        y = np.sin(2 * t) / 2
    elif shape == "spiral":
        # Radius grows from 0.2 to 1.0 over 3 full rotations
        r = np.linspace(0.2, 1.0, n_points)
        x = r * np.cos(3 * t)
        y = r * np.sin(3 * t)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return np.column_stack((x, y))


def add_tremor(path, frequency=10.0, amplitude=0.15, sample_rate=300):
    """
    Add simulated hand tremor to a clean path.

    Tremor is modeled as:
      1. A sinusoidal oscillation at the given frequency (the main tremor component)
      2. Small gaussian noise on top (random hand jitter)

    Real Parkinsonian tremor oscillates at 4-12 Hz. We use higher frequencies
    (up to 60 Hz) in experiments to show the filter's behavior across a range.

    Args:
        path:        (N, 2) array of clean (x, y) coordinates
        frequency:   tremor oscillation frequency in Hz
        amplitude:   peak displacement of the tremor (in path units)
        sample_rate: number of samples per second (used to build the time axis)
    """
    n = len(path)
    # Build a time axis so the oscillation frequency is correct
    t = np.linspace(0, n / sample_rate, n)

    # Sinusoidal tremor — x and y are phase-shifted by π/4 so the tremor
    # oscillates diagonally rather than purely horizontally
    tremor_x = amplitude * np.sin(2 * np.pi * frequency * t)
    tremor_y = amplitude * np.cos(2 * np.pi * frequency * t + np.pi / 4)

    # Small random noise to make the tremor look organic, not perfectly sinusoidal
    noise_x = np.random.normal(0, amplitude * 0.15, n)
    noise_y = np.random.normal(0, amplitude * 0.15, n)

    noisy = path.copy()
    noisy[:, 0] += tremor_x + noise_x
    noisy[:, 1] += tremor_y + noise_y

    return noisy
