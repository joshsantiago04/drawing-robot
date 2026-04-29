import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec



def plot_comparison(true_path, noisy_path, filtered_path, title="Tremor Suppression Results"):
    """Side-by-side comparison of true, noisy, and filtered paths."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    datasets = [
        (true_path,     "Ground Truth",   "green"),
        (noisy_path,    "With Tremor",    "red"),
        (filtered_path, "Kalman Filtered","blue"),
    ]

    for ax, (path, label, color) in zip(axes, datasets):
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=0.8, alpha=0.8)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error(true_path, noisy_path, filtered_path):
    """Plot position error over time for noisy vs filtered."""
    noisy_error = np.linalg.norm(noisy_path - true_path, axis=1)
    filtered_error = np.linalg.norm(filtered_path - true_path, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(noisy_error,    label=f"Tremor error    (mean={noisy_error.mean():.3f})",
            color="red",  alpha=0.7)
    ax.plot(filtered_error, label=f"Filtered error  (mean={filtered_error.mean():.3f})",
            color="blue", alpha=0.9)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Position error (units)")
    ax.set_title("Position Error: Tremor vs. Kalman Filtered")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def animate_arm(arm, filtered_path, true_path, interval=20, step=3, title="Robot Arm Drawing (Kalman Filtered)"):
    """
    Animate the robot arm following the filtered path.
    step: only render every Nth frame to speed up the animation.
    """
    joint_angles, tip_positions = arm.follow_path(filtered_path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Draw the intended path faintly in the background
    scaled_true = np.column_stack((true_path[:, 0] * 0.5 + 0.9, true_path[:, 1] * 0.5))
    ax.plot(scaled_true[:, 0], scaled_true[:, 1],
            color="green", linewidth=1, alpha=0.3, linestyle="--", label="Intended path")

    arm_line,  = ax.plot([], [], "o-", color="steelblue",  linewidth=3, markersize=8)
    trace_line, = ax.plot([], [], "-", color="royalblue", linewidth=1.2, alpha=0.7)
    ax.legend(loc="upper right")

    trace_x, trace_y = [], []
    frames = range(0, len(joint_angles), step)

    def init():
        arm_line.set_data([], [])
        trace_line.set_data([], [])
        return arm_line, trace_line

    def update(i):
        theta1, theta2 = joint_angles[i]
        base, elbow, tip = arm.forward_kinematics(theta1, theta2)

        arm_line.set_data([base[0], elbow[0], tip[0]],
                          [base[1], elbow[1], tip[1]])

        trace_x.append(tip[0])
        trace_y.append(tip[1])
        trace_line.set_data(trace_x, trace_y)

        return arm_line, trace_line

    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init,
        interval=interval, blit=True
    )
    return fig, ani
