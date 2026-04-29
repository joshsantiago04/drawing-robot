# robot.py
# 2-DOF planar robot arm model with inverse and forward kinematics.
#
# The arm has two rigid segments connected by joints:
#   - Shoulder joint (theta1): rotates the first segment relative to the base
#   - Elbow joint   (theta2): rotates the second segment relative to the first
#
# The base is fixed at the origin. The tip holds the drawing pen.
#
# Two kinematics directions:
#   Forward  (easy): given joint angles → where is the tip?
#   Inverse  (hard): given a tip position → what angles do the joints need?
#
# Inverse kinematics is solved geometrically using the law of cosines.

import numpy as np
from pid import JointPIDController


class PlanarArm:
    """2-DOF planar robot arm with IK, FK, and PID-controlled path following."""

    def __init__(self, l1=1.0, l2=0.8):
        self.l1 = l1   # length of upper arm segment (shoulder → elbow)
        self.l2 = l2   # length of forearm segment   (elbow → tip)
        self.max_reach = l1 + l2  # furthest the tip can reach from the base

    def inverse_kinematics(self, x, y):
        """
        Given a target tip position (x, y), return the required joint angles
        (theta1, theta2) using the geometric law-of-cosines approach.

        If the target is outside the arm's reach it is clamped to the boundary.
        """
        dist = np.sqrt(x**2 + y**2)

        # Clamp target to just inside max reach to avoid arccos domain errors
        if dist > self.max_reach:
            scale = self.max_reach * 0.99 / dist
            x, y  = x * scale, y * scale
            dist  = np.sqrt(x**2 + y**2)

        # Law of cosines: find elbow angle (theta2)
        # cos(θ2) = (d² - l1² - l2²) / (2·l1·l2)
        cos_theta2 = (dist**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)   # guard against floating-point drift
        theta2 = np.arccos(cos_theta2)  # elbow-up solution (always positive)

        # Shoulder angle (theta1): angle to target minus elbow offset
        alpha = np.arctan2(y, x)
        beta  = np.arccos(
            np.clip((dist**2 + self.l1**2 - self.l2**2) / (2 * dist * self.l1), -1.0, 1.0)
        )
        theta1 = alpha - beta

        return theta1, theta2

    def forward_kinematics(self, theta1, theta2):
        """
        Given joint angles, return (base, elbow, tip) positions.
        Used to draw the arm visually and to compute the pen trace.
        """
        # Elbow is at distance l1 from base, at angle theta1
        elbow_x = self.l1 * np.cos(theta1)
        elbow_y = self.l1 * np.sin(theta1)

        # Tip is at distance l2 from elbow, at combined angle theta1 + theta2
        tip_x = elbow_x + self.l2 * np.cos(theta1 + theta2)
        tip_y = elbow_y + self.l2 * np.sin(theta1 + theta2)

        return (0.0, 0.0), (elbow_x, elbow_y), (tip_x, tip_y)

    def follow_path(self, path):
        """
        Follow a path by solving IK at each waypoint and jumping directly
        to the resulting joint angles (no motor dynamics).

        The path is scaled and offset so the drawing sits in the arm's
        natural working range and the elbow visibly flexes during motion.
        """
        joint_angles  = []
        tip_positions = []

        for x, y in path:
            # Scale path to 50% size and offset to the right of the arm base
            # so the elbow must flex to reach all parts of the drawing
            tx, ty = x * 0.5 + 0.9, y * 0.5
            angles = self.inverse_kinematics(tx, ty)
            joint_angles.append(angles)
            _, _, tip = self.forward_kinematics(*angles)
            tip_positions.append(tip)

        return np.array(joint_angles), np.array(tip_positions)

    def follow_path_pid(self, path, kp=8.0, ki=0.05, kd=0.4, steps_per_waypoint=20):
        """
        Follow a path using PID-controlled joints for smooth, realistic motion.

        Unlike follow_path (which teleports), this drives each joint toward
        the IK target incrementally — simulating real motor behavior.
        The tradeoff is slightly higher position error due to tracking lag.
        """
        # First compute the IK target for every waypoint
        target_angles = []
        for x, y in path:
            tx, ty = x * 0.5 + 0.9, y * 0.5
            target_angles.append(self.inverse_kinematics(tx, ty))

        # Then use PID controllers to smoothly drive the joints
        pid = JointPIDController(kp=kp, ki=ki, kd=kd,
                                 steps_per_waypoint=steps_per_waypoint)
        angle_history = pid.follow_angles(target_angles)

        tip_positions = np.array([
            self.forward_kinematics(t1, t2)[2]
            for t1, t2 in angle_history
        ])

        return angle_history, tip_positions
