# Drawing Robot — Tremor Suppression Simulation

**CPS_S 483 Final Project**

A Python simulation of a robotic drawing assistant that filters involuntary hand tremor using a Kalman filter, then replays the smoothed path on a 2-DOF planar robot arm.

## Problem

People with motor disorders (Parkinson's, essential tremor) have involuntary hand shaking that prevents activities like drawing. This system captures raw hand movement, strips out the tremor using signal filtering, and physically draws what the user *intended* — not what their shaky hand actually produced.

## How It Works

```
Raw input (shaky) → Kalman Filter → Cleaned path → Inverse Kinematics → PID Control → Robot Arm
```

1. **Tremor simulation** — generates a clean path (circle, figure-8, spiral) then adds sinusoidal + gaussian noise mimicking Parkinsonian tremor (4–60 Hz)
2. **Kalman filter** — tracks position + velocity as state, predicts next position from physics, blends prediction with noisy measurement
3. **Inverse kinematics** — uses the law of cosines to find joint angles needed to reach each target position
4. **PID control** — drives each joint smoothly toward its target angle instead of teleporting

## Setup

```bash
python3 -m venv venv
venv/bin/pip install numpy matplotlib customtkinter
```

## Running

### Interactive GUI (recommended for demo)
```bash
venv/bin/python gui.py
```
Draw in the left panel, click **Run Robot** to filter and animate.

### Animations for all shapes
```bash
venv/bin/python main.py
```

### Generate all experiment result plots
```bash
venv/bin/python experiments.py
```
Saves all plots to `results/`.

## File Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point — runs the full pipeline for all shapes |
| `tremor.py` | Generates clean paths and adds simulated tremor |
| `kalman.py` | 2D Kalman filter (state: position + velocity) |
| `robot.py` | 2-DOF planar arm — inverse/forward kinematics |
| `pid.py` | PID controllers for smooth joint actuation |
| `visualize.py` | Matplotlib plots and robot arm animation |
| `experiments.py` | Systematic experiments across frequencies, amplitudes, shapes |
| `gui.py` | Interactive GUI — draw input, see filtered output and arm animation |

## Results

All experiment plots are saved to `results/` after running `experiments.py`:

| File | What it shows |
|------|--------------|
| `comparison_*.png` | Ground truth vs. tremor vs. Kalman filtered |
| `error_plot_*.png` | Position error over time |
| `results_frequency.png` | Filter performance at 10 / 25 / 60 Hz tremor |
| `results_amplitude.png` | Filter performance at low / medium / high amplitude |
| `results_summary.png` | Bar chart comparing all configurations |
| `results_fft.png` | Frequency spectrum — tremor spike visible and suppressed |
| `results_pid.png` | Instant IK vs. PID-controlled arm tracking |
| `results_shapes.png` | Filter across circle, figure-8, and spiral shapes |

## Key Findings

- Higher tremor frequency → easier for the Kalman filter to suppress (~70% reduction at 60 Hz)
- Lower tremor frequency → harder to suppress (~5% at 10 Hz) because slow tremor looks like intentional motion
- PID control introduces slight tracking lag vs. instant IK, but produces physically realistic arm motion
