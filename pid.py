# pid.py
# PID (Proportional-Integral-Derivative) controller for joint angle control.
#
# In a real robot, each joint has a motor. A PID controller tells that motor
# how hard to push based on how far the joint is from its target angle:
#
#   P (Proportional) — push harder the further you are from the target
#   I (Integral)     — correct for persistent steady-state error over time
#   D (Derivative)   — dampen oscillation by reacting to how fast error changes
#
# This simulates what the motor controller firmware would run in hardware.

import numpy as np


class PIDController:
    """Single-axis PID controller for one joint."""

    def __init__(self, kp=5.0, ki=0.1, kd=0.5, dt=0.01, output_limit=None):
        # Gain terms — tuned empirically for smooth arm tracking
        self.kp = kp   # proportional gain
        self.ki = ki   # integral gain
        self.kd = kd   # derivative gain
        self.dt = dt   # time step between control updates

        # Optional clamp to prevent unrealistically large control signals
        self.output_limit = output_limit

        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Clear accumulated state — call this before tracking a new path."""
        self._integral   = 0.0
        self._prev_error = 0.0

    def step(self, current, target):
        """
        Compute one control output given the current and target joint angle.
        Returns the correction to apply this time step.
        """
        error = target - current

        # Accumulate error over time (integral term)
        self._integral += error * self.dt

        # Rate of change of error (derivative term — reduces overshoot)
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        return output


class JointPIDController:
    """
    Two independent PID controllers for the two joints of a 2-DOF arm.

    Instead of jumping directly to the IK solution (which teleports the arm),
    this drives each joint incrementally toward the target angle, simulating
    real motor dynamics with smooth acceleration and deceleration.
    """

    def __init__(self, kp=8.0, ki=0.05, kd=0.4, dt=0.01, steps_per_waypoint=20):
        # One PID per joint (shoulder and elbow)
        self.pid1 = PIDController(kp, ki, kd, dt, output_limit=2.0)
        self.pid2 = PIDController(kp, ki, kd, dt, output_limit=2.0)
        self.dt = dt
        # Number of control sub-steps taken between each path waypoint
        self.steps_per_waypoint = steps_per_waypoint

    def follow_angles(self, target_angles):
        """
        Drive the arm through a sequence of target joint angle pairs.

        For each waypoint, runs `steps_per_waypoint` PID iterations, letting
        the joints converge toward the target before moving to the next one.

        Returns the full joint angle history (one entry per sub-step).
        """
        # Start from the first target position
        theta1, theta2 = target_angles[0]
        self.pid1.reset()
        self.pid2.reset()

        history = []

        for t1_target, t2_target in target_angles:
            for _ in range(self.steps_per_waypoint):
                # Each joint independently tracks its target angle
                theta1 += self.pid1.step(theta1, t1_target) * self.dt
                theta2 += self.pid2.step(theta2, t2_target) * self.dt
                history.append((theta1, theta2))

        return np.array(history)
