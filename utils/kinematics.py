"""
UR10e Kinematics Module

Forward and inverse kinematics calculations for UR10e robotic arm.
Supports both analytical and numerical solutions.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass


@dataclass
class DHParameters:
    """Denavit-Hartenberg parameters for UR10e"""
    a: np.ndarray      # Link lengths
    d: np.ndarray      # Link offsets
    alpha: np.ndarray  # Twist angles

    def __init__(self):
        # UR10e DH parameters (in meters and radians)
        self.a = np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0])
        self.d = np.array([0.1273, 0.0, 0.0, 0.1639, 0.1157, 0.0922])
        self.alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])


@dataclass
class UR10eSpecs:
    """UR10e robot specifications"""
    joint_limits: np.ndarray
    torque_limits: np.ndarray
    max_velocity: np.ndarray
    max_acceleration: np.ndarray

    def __init__(self):
        # Joint limits in radians
        self.joint_limits = np.array([
            [-2.0*np.pi, 2.0*np.pi],  # Shoulder pan
            [-np.pi, np.pi],          # Shoulder lift
            [-np.pi, np.pi],          # Elbow
            [-2.0*np.pi, 2.0*np.pi],  # Wrist 1
            [-2.0*np.pi, 2.0*np.pi],  # Wrist 2
            [-2.0*np.pi, 2.0*np.pi]   # Wrist 3
        ])

        # Torque limits in N⋅m
        self.torque_limits = np.array([330.0, 330.0, 330.0, 54.0, 54.0, 54.0])

        # Maximum velocities in rad/s
        self.max_velocity = np.array([2.16, 2.16, 2.16, 3.15, 3.15, 3.15])

        # Maximum accelerations in rad/s²
        self.max_acceleration = np.array([3.5, 3.5, 3.5, 6.0, 6.0, 6.0])


class UR10eKinematics:
    """
    UR10e kinematics solver

    Provides forward and inverse kinematics for UR10e robotic arm.
    Supports both NumPy and PyTorch operations.
    """

    def __init__(self):
        """Initialize UR10e kinematics solver"""
        self.dh = DHParameters()
        self.specs = UR10eSpecs()

        # Precompute some constants
        self._setup_kinematics()

    def _setup_kinematics(self):
        """Setup precomputed kinematics constants"""
        # Link lengths
        self.L1 = 0.1273  # Base to shoulder
        self.L2 = 0.612   # Shoulder to elbow
        self.L3 = 0.5723  # Elbow to wrist1
        self.L4 = 0.1639  # Wrist1 to wrist2
        self.L5 = 0.1157  # Wrist2 to wrist3
        self.L6 = 0.0922  # Wrist3 to end-effector

    def forward_kinematics(self, joint_angles: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute forward kinematics

        Args:
            joint_angles: 6D joint configuration [q1, q2, q3, q4, q5, q6]

        Returns:
            4x4 homogeneous transformation matrix or 3D position
        """
        if isinstance(joint_angles, torch.Tensor):
            return self._forward_kinematics_torch(joint_angles)
        else:
            return self._forward_kinematics_numpy(joint_angles)

    def _forward_kinematics_numpy(self, q: np.ndarray) -> np.ndarray:
        """Forward kinematics using NumPy"""
        q = np.asarray(q)
        if q.ndim == 1:
            q = q.reshape(6)

        # DH transformation matrices
        def dh_transform(a, d, alpha, theta):
            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(theta), np.sin(theta)
            return np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])

        # Compute individual transformations
        T01 = dh_transform(self.dh.a[0], self.dh.d[0], self.dh.alpha[0], q[0])
        T12 = dh_transform(self.dh.a[1], self.dh.d[1], self.dh.alpha[1], q[1])
        T23 = dh_transform(self.dh.a[2], self.dh.d[2], self.dh.alpha[2], q[2])
        T34 = dh_transform(self.dh.a[3], self.dh.d[3], self.dh.alpha[3], q[3])
        T45 = dh_transform(self.dh.a[4], self.dh.d[4], self.dh.alpha[4], q[4])
        T56 = dh_transform(self.dh.a[5], self.dh.d[5], self.dh.alpha[5], q[5])

        # Compute total transformation
        T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56

        return T06

    def _forward_kinematics_torch(self, q: torch.Tensor) -> torch.Tensor:
        """Forward kinematics using PyTorch (batch processing)"""
        if q.ndim == 1:
            q = q.unsqueeze(0)  # Add batch dimension

        batch_size = q.shape[0]
        device = q.device

        # Initialize transformation matrices
        T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # DH parameters as tensors
        a = torch.tensor(self.dh.a, device=device)
        d = torch.tensor(self.dh.d, device=device)
        alpha = torch.tensor(self.dh.alpha, device=device)

        # Compute transformations for each joint
        for i in range(6):
            ca, sa = torch.cos(alpha[i]), torch.sin(alpha[i])
            ct, st = torch.cos(q[:, i]), torch.sin(q[:, i])

            Ti = torch.stack([
                torch.stack([ct, -st*ca, st*sa, a[i]*ct], dim=1),
                torch.stack([st, ct*ca, -ct*sa, a[i]*st], dim=1),
                torch.stack([torch.zeros_like(ct), sa*torch.ones_like(ct), ca*torch.ones_like(ct), d[i]*torch.ones_like(ct)], dim=1),
                torch.stack([torch.zeros_like(ct), torch.zeros_like(ct), torch.zeros_like(ct), torch.ones_like(ct)], dim=1)
            ], dim=1)

            T = torch.bmm(T, Ti)

        # Remove batch dimension if single input
        if batch_size == 1:
            return T[0]

        return T

    def get_tcp_position(self, joint_angles: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Get TCP (Tool Center Point) position from joint angles

        Args:
            joint_angles: 6D joint configuration

        Returns:
            3D position [x, y, z]
        """
        T = self.forward_kinematics(joint_angles)

        if isinstance(T, torch.Tensor):
            return T[:3, 3]
        else:
            return T[:3, 3]

    def inverse_kinematics(self,
                          target_position: Union[np.ndarray, torch.Tensor],
                          target_orientation: Optional[Union[np.ndarray, torch.Tensor]] = None,
                          initial_guess: Optional[Union[np.ndarray, torch.Tensor]] = None,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute inverse kinematics using numerical optimization

        Args:
            target_position: Target 3D position [x, y, z]
            target_orientation: Target orientation (optional)
            initial_guess: Initial joint configuration guess
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance

        Returns:
            6D joint configuration or None if no solution found
        """
        if isinstance(target_position, torch.Tensor):
            return self._inverse_kinematics_torch(target_position, target_orientation,
                                                  initial_guess, max_iterations, tolerance)
        else:
            return self._inverse_kinematics_numpy(target_position, target_orientation,
                                                   initial_guess, max_iterations, tolerance)

    def _inverse_kinematics_numpy(self, target_pos, target_orient=None, initial_guess=None,
                                 max_iterations=100, tolerance=1e-6):
        """Inverse kinematics using NumPy (Newton-Raphson method)"""
        target_pos = np.asarray(target_pos)

        # Use center pose as initial guess if not provided
        if initial_guess is None:
            q = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
        else:
            q = np.asarray(initial_guess).copy()

        for iteration in range(max_iterations):
            # Current position
            current_pos = self.get_tcp_position(q)
            pos_error = target_pos - current_pos

            # Check convergence
            if np.linalg.norm(pos_error) < tolerance:
                return q

            # Compute Jacobian
            J = self._compute_jacobian_numpy(q)

            # Update joint angles
            delta_q = np.linalg.pinv(J) @ pos_error
            q += delta_q * 0.1  # Step size for stability

            # Apply joint limits
            q = np.clip(q, self.specs.joint_limits[:, 0], self.specs.joint_limits[:, 1])

        print(f"⚠️ IK did not converge after {max_iterations} iterations, error: {np.linalg.norm(pos_error):.6f}")
        return q

    def _inverse_kinematics_torch(self, target_pos, target_orient=None, initial_guess=None,
                                 max_iterations=100, tolerance=1e-6):
        """Inverse kinematics using PyTorch (batch processing)"""
        if target_pos.ndim == 1:
            target_pos = target_pos.unsqueeze(0)

        batch_size = target_pos.shape[0]
        device = target_pos.device

        # Use center pose as initial guess if not provided
        if initial_guess is None:
            q = torch.tensor([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0],
                           device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        else:
            q = initial_guess.clone()

        for iteration in range(max_iterations):
            # Current position
            current_pos = self.get_tcp_position(q)
            pos_error = target_pos - current_pos

            # Check convergence
            errors = torch.norm(pos_error, dim=1)
            if torch.all(errors < tolerance):
                return q[0] if batch_size == 1 else q

            # Compute Jacobian
            J = self._compute_jacobian_torch(q)

            # Update joint angles
            delta_q = torch.linalg.pinv(J) @ pos_error.unsqueeze(-1)
            q += delta_q.squeeze(-1) * 0.1  # Step size for stability

            # Apply joint limits
            limits_lower = torch.tensor(self.specs.joint_limits[:, 0], device=device)
            limits_upper = torch.tensor(self.specs.joint_limits[:, 1], device=device)
            q = torch.clamp(q, limits_lower, limits_upper)

        print(f"⚠️ IK did not converge after {max_iterations} iterations")
        return q[0] if batch_size == 1 else q

    def _compute_jacobian_numpy(self, q):
        """Compute Jacobian matrix using NumPy"""
        J = np.zeros((3, 6))
        delta = 1e-6

        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += delta

            pos_plus = self.get_tcp_position(q_plus)
            pos_original = self.get_tcp_position(q)

            J[:, i] = (pos_plus - pos_original) / delta

        return J

    def _compute_jacobian_torch(self, q):
        """Compute Jacobian matrix using PyTorch"""
        batch_size = q.shape[0]
        device = q.device
        delta = 1e-6

        J = torch.zeros((batch_size, 3, 6), device=device)
        pos_original = self.get_tcp_position(q)

        for i in range(6):
            q_plus = q.clone()
            q_plus[:, i] += delta

            pos_plus = self.get_tcp_position(q_plus)
            J[:, :, i] = (pos_plus - pos_original).unsqueeze(-1) / delta

        return J

    def check_reachability(self, position: Union[np.ndarray, torch.Tensor]) -> bool:
        """
        Check if a position is reachable by the robot

        Args:
            position: 3D position to check

        Returns:
            True if reachable, False otherwise
        """
        if isinstance(position, torch.Tensor):
            pos_np = position.cpu().numpy()
        else:
            pos_np = np.asarray(position)

        # Calculate distance from base
        distance = np.linalg.norm(pos_np)

        # UR10e reach envelope
        min_reach = self.L1  # Minimum reach
        max_reach = self.L2 + self.L3 + self.L4 + self.L5 + self.L6  # Maximum reach

        return min_reach <= distance <= max_reach

    def joint_angles_to_tcp_pose(self, joint_angles: Union[np.ndarray, torch.Tensor]) -> dict:
        """
        Convert joint angles to full TCP pose (position + orientation)

        Args:
            joint_angles: 6D joint configuration

        Returns:
            Dictionary with position and orientation
        """
        T = self.forward_kinematics(joint_angles)

        if isinstance(T, torch.Tensor):
            T_np = T.cpu().numpy()
        else:
            T_np = T

        # Extract position and orientation
        position = T_np[:3, 3]

        # Extract rotation matrix and convert to Euler angles
        R = T_np[:3, :3]

        # Compute Euler angles (ZYZ convention)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return {
            'position': position,
            'orientation': np.array([x, y, z]),
            'rotation_matrix': R,
            'homogeneous_matrix': T_np
        }

    def get_manipulability(self, joint_angles: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute manipulability measure for given joint configuration

        Args:
            joint_angles: 6D joint configuration

        Returns:
            Manipulability measure (0 to 1, higher is better)
        """
        J = self._compute_jacobian_numpy(joint_angles) if isinstance(joint_angles, np.ndarray) else \
            self._compute_jacobian_torch(joint_angles.unsqueeze(0))[0].cpu().numpy()

        # Yoshikawa manipulability measure
        manipulability = np.sqrt(np.linalg.det(J @ J.T))

        return manipulability


def test_kinematics():
    """Test UR10e kinematics functions"""
    print("🧪 Testing UR10e Kinematics")

    kinematics = UR10eKinematics()

    # Test joint configuration
    q = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])

    print(f"📐 Test joint configuration: {q}")

    # Forward kinematics
    tcp_pos = kinematics.get_tcp_position(q)
    print(f"🎯 TCP position: {tcp_pos}")

    # Full pose
    pose = kinematics.joint_angles_to_tcp_pose(q)
    print(f"📍 TCP pose - Position: {pose['position']}")
    print(f"📐 TCP pose - Orientation (rad): {pose['orientation']}")

    # Inverse kinematics
    recovered_q = kinematics.inverse_kinematics(tcp_pos, initial_guess=q)
    print(f"🔄 Recovered joint angles: {recovered_q}")
    print(f"📏 IK error: {np.linalg.norm(q - recovered_q):.6f}")

    # Reachability test
    reachable_pos = np.array([0.5, 0.3, 0.4])
    unreachable_pos = np.array([2.0, 2.0, 2.0])

    print(f"✅ Reachable [{reachable_pos}]: {kinematics.check_reachability(reachable_pos)}")
    print(f"❌ Unreachable [{unreachable_pos}]: {kinematics.check_reachability(unreachable_pos)}")

    # Manipulability
    manip = kinematics.get_manipulability(q)
    print(f"🎮 Manipulability: {manip:.4f}")

    print("✅ Kinematics test completed")


if __name__ == "__main__":
    test_kinematics()