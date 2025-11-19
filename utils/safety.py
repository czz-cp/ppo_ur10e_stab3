"""
Safety Module for UR10e

Safety monitoring, collision detection, and emergency stop functionality.
Protects robot and environment during RL training and execution.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class SafetyLevel(Enum):
    """Safety severity levels"""
    NORMAL = 0      # Normal operation
    WARNING = 1     # Warning condition
    CRITICAL = 2    # Critical condition
    EMERGENCY = 3   # Emergency stop required


@dataclass
class SafetyParameters:
    """Safety system parameters"""
    # Joint limits (with safety margin)
    joint_position_margin: float = 0.05      # radians
    joint_velocity_limit: float = 3.0        # rad/s
    joint_acceleration_limit: float = 5.0    # rad/s²

    # Torque limits (with safety factor)
    torque_safety_factor: float = 0.8        # 80% of max torque
    max_instantaneous_torque: float = 500.0   # N⋅m

    # Workspace limits
    workspace_radius: float = 1.2            # meters
    min_height: float = 0.05                 # meters
    max_height: float = 1.5                  # meters

    # Collision and proximity
    collision_threshold: float = 0.02        # meters
    approach_warning_distance: float = 0.1   # meters
    self_collision_threshold: float = 0.05   # meters

    # Emergency conditions
    max_velocity_error: float = 0.5          # rad/s
    max_position_error: float = 0.1          # meters
    max_control_deviation: float = 0.2       # radians

    # Monitoring
    history_window: int = 100                # Time steps for history


@dataclass
class SafetyEvent:
    """Safety event record"""
    timestamp: float
    level: SafetyLevel
    message: str
    joint_states: Optional[np.ndarray] = None
    torques: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    resolved: bool = False


class SafetyMonitor:
    """
    Safety monitoring system for UR10e

    Monitors joint states, torques, positions, and detects dangerous conditions.
    Provides emergency stop functionality and safety event logging.
    """

    def __init__(self, ur10e_specs: Dict[str, Any], params: Optional[SafetyParameters] = None):
        """
        Initialize safety monitor

        Args:
            ur10e_specs: UR10e robot specifications
            params: Safety parameters (uses defaults if None)
        """
        self.specs = ur10e_specs
        self.params = params or SafetyParameters()

        # Safety state
        self.current_level = SafetyLevel.NORMAL
        self.emergency_stop_active = False
        self.last_check_time = 0.0

        # Event history
        self.safety_events: List[SafetyEvent] = []
        self.state_history: List[Dict[str, Any]] = []

        # Monitoring data
        self.last_joint_positions: Optional[np.ndarray] = None
        self.last_joint_velocities: Optional[np.ndarray] = None
        self.last_check_time = 0.0

        print("✅ Safety monitor initialized")
        print(f"   Joint position margin: {self.params.joint_position_margin:.3f} rad")
        print(f"   Torque safety factor: {self.params.torque_safety_factor:.1%}")
        print(f"   Workspace radius: {self.params.workspace_radius:.2f} m")

    def check_safety(self,
                    joint_positions: np.ndarray,
                    joint_velocities: np.ndarray,
                    joint_torques: np.ndarray,
                    tcp_position: Optional[np.ndarray] = None,
                    timestamp: Optional[float] = None) -> SafetyLevel:
        """
        Perform comprehensive safety check

        Args:
            joint_positions: Current joint positions
            joint_velocities: Current joint velocities
            joint_torques: Current joint torques
            tcp_position: Current TCP position (optional)
            timestamp: Current timestamp (optional)

        Returns:
            Current safety level
        """
        if timestamp is None:
            timestamp = self._get_current_time()

        # Initialize safety level
        max_level = SafetyLevel.NORMAL

        # Store state history
        self._store_state_history(joint_positions, joint_velocities, joint_torques, tcp_position, timestamp)

        # Check joint limits
        joint_level = self._check_joint_limits(joint_positions)
        max_level = SafetyLevel(max(max_level.value, joint_level.value))

        # Check velocity limits
        velocity_level = self._check_velocity_limits(joint_velocities)
        max_level = SafetyLevel(max(max_level.value, velocity_level.value))

        # Check torque limits
        torque_level = self._check_torque_limits(joint_torques)
        max_level = SafetyLevel(max(max_level.value, torque_level.value))

        # Check workspace limits
        if tcp_position is not None:
            workspace_level = self._check_workspace_limits(tcp_position)
            max_level = SafetyLevel(max(max_level.value, workspace_level.value))

        # Check for unexpected behavior
        behavior_level = self._check_behavior_anomalies(joint_positions, joint_velocities)
        max_level = SafetyLevel(max(max_level.value, behavior_level.value))

        # Update safety state
        self.current_level = max_level

        # Handle emergency conditions
        if max_level == SafetyLevel.EMERGENCY:
            self._trigger_emergency_stop()

        # Log events
        if max_level != SafetyLevel.NORMAL:
            self._log_safety_event(max_level, self._generate_safety_message(max_level),
                                 joint_positions, joint_torques, tcp_position, timestamp)

        return max_level

    def _check_joint_limits(self, joint_positions: np.ndarray) -> SafetyLevel:
        """Check joint position limits"""
        joint_limits = self.specs['joint_limits']
        margin = self.params.joint_position_margin

        for i, pos in enumerate(joint_positions):
            if pos < joint_limits[i, 0] - margin or pos > joint_limits[i, 1] + margin:
                return SafetyLevel.EMERGENCY
            elif pos < joint_limits[i, 0] - margin/2 or pos > joint_limits[i, 1] + margin/2:
                return SafetyLevel.CRITICAL
            elif pos < joint_limits[i, 0] - margin/4 or pos > joint_limits[i, 1] + margin/4:
                return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def _check_velocity_limits(self, joint_velocities: np.ndarray) -> SafetyLevel:
        """Check joint velocity limits"""
        max_vel = self.params.joint_velocity_limit

        for i, vel in enumerate(joint_velocities):
            if abs(vel) > max_vel * 1.5:
                return SafetyLevel.EMERGENCY
            elif abs(vel) > max_vel * 1.2:
                return SafetyLevel.CRITICAL
            elif abs(vel) > max_vel:
                return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def _check_torque_limits(self, joint_torques: np.ndarray) -> SafetyLevel:
        """Check torque limits"""
        torque_limits = self.specs['torque_limits']
        safety_factor = self.params.torque_safety_factor

        for i, torque in enumerate(joint_torques):
            safe_limit = torque_limits[i] * safety_factor

            if abs(torque) > safe_limit * 1.5:
                return SafetyLevel.EMERGENCY
            elif abs(torque) > safe_limit * 1.2:
                return SafetyLevel.CRITICAL
            elif abs(torque) > safe_limit:
                return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def _check_workspace_limits(self, tcp_position: np.ndarray) -> SafetyLevel:
        """Check workspace limits"""
        distance = np.linalg.norm(tcp_position[:2])  # Distance from Z-axis
        height = tcp_position[2]

        if distance > self.params.workspace_radius * 1.5 or height > self.params.max_height * 1.5:
            return SafetyLevel.EMERGENCY
        elif distance > self.params.workspace_radius * 1.2 or height > self.params.max_height * 1.2:
            return SafetyLevel.CRITICAL
        elif distance > self.params.workspace_radius or height > self.params.max_height:
            return SafetyLevel.WARNING
        elif height < self.params.min_height:
            return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def _check_behavior_anomalies(self, joint_positions: np.ndarray, joint_velocities: np.ndarray) -> SafetyLevel:
        """Check for unexpected behavior patterns"""
        # Check for sudden position changes
        if self.last_joint_positions is not None:
            position_diff = np.linalg.norm(joint_positions - self.last_joint_positions)
            if position_diff > self.params.max_position_error:
                return SafetyLevel.EMERGENCY
            elif position_diff > self.params.max_position_error * 0.5:
                return SafetyLevel.WARNING

        # Check for velocity anomalies
        if self.last_joint_velocities is not None:
            velocity_diff = np.linalg.norm(joint_velocities - self.last_joint_velocities)
            if velocity_diff > self.params.max_velocity_error:
                return SafetyLevel.CRITICAL
            elif velocity_diff > self.params.max_velocity_error * 0.5:
                return SafetyLevel.WARNING

        # Update stored values
        self.last_joint_positions = joint_positions.copy()
        self.last_joint_velocities = joint_velocities.copy()

        return SafetyLevel.NORMAL

    def _trigger_emergency_stop(self):
        """Trigger emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self._log_safety_event(SafetyLevel.EMERGENCY, "Emergency stop triggered")
            print("🚨 EMERGENCY STOP TRIGGERED")

    def clear_emergency_stop(self):
        """Clear emergency stop condition"""
        if self.emergency_stop_active:
            self.emergency_stop_active = False
            self._log_safety_event(SafetyLevel.NORMAL, "Emergency stop cleared")
            print("✅ Emergency stop cleared")

    def is_safe_to_operate(self) -> bool:
        """Check if it's safe to operate the robot"""
        return self.current_level != SafetyLevel.EMERGENCY and not self.emergency_stop_active

    def get_safe_torques(self, desired_torques: np.ndarray) -> np.ndarray:
        """
        Apply safety limits to desired torques

        Args:
            desired_torques: Desired joint torques

        Returns:
            Safe torques within limits
        """
        torque_limits = self.specs['torque_limits']
        safety_factor = self.params.torque_safety_factor

        safe_torques = np.zeros_like(desired_torques)

        for i in range(len(desired_torques)):
            safe_limit = torque_limits[i] * safety_factor
            safe_torques[i] = np.clip(desired_torques[i], -safe_limit, safe_limit)

        return safe_torques

    def check_self_collision(self, joint_positions: np.ndarray) -> SafetyLevel:
        """
        Check for self-collision (simplified)

        Args:
            joint_positions: Current joint configuration

        Returns:
            Safety level based on collision risk
        """
        # This is a simplified self-collision check
        # In practice, you would use proper collision detection with the robot model

        # Check for configurations that might cause self-collision
        # (This is a very simplified heuristic)

        # Example: Check if wrist 1 is too close to base
        if joint_positions[1] > np.pi/2 and joint_positions[2] > np.pi/2:
            return SafetyLevel.WARNING

        # Example: Check for singular configurations
        if abs(joint_positions[1] + joint_positions[2] + joint_positions[3]) < 0.1:
            return SafetyLevel.WARNING

        return SafetyLevel.NORMAL

    def _store_state_history(self, joint_positions, joint_velocities, joint_torques, tcp_position, timestamp):
        """Store current state in history"""
        state = {
            'timestamp': timestamp,
            'joint_positions': joint_positions.copy(),
            'joint_velocities': joint_velocities.copy(),
            'joint_torques': joint_torques.copy(),
            'tcp_position': tcp_position.copy() if tcp_position is not None else None
        }

        self.state_history.append(state)

        # Limit history size
        if len(self.state_history) > self.params.history_window:
            self.state_history.pop(0)

    def _log_safety_event(self, level: SafetyLevel, message: str,
                         joint_positions=None, joint_torques=None, tcp_position=None, timestamp=None):
        """Log safety event"""
        if timestamp is None:
            timestamp = self._get_current_time()

        event = SafetyEvent(
            timestamp=timestamp,
            level=level,
            message=message,
            joint_positions=joint_positions.copy() if joint_positions is not None else None,
            torques=joint_torques.copy() if joint_torques is not None else None,
            position=tcp_position.copy() if tcp_position is not None else None
        )

        self.safety_events.append(event)

        # Limit event history
        if len(self.safety_events) > 1000:
            self.safety_events.pop(0)

    def _generate_safety_message(self, level: SafetyLevel) -> str:
        """Generate safety level message"""
        messages = {
            SafetyLevel.NORMAL: "Normal operation",
            SafetyLevel.WARNING: "Warning condition detected",
            SafetyLevel.CRITICAL: "Critical condition - attention required",
            SafetyLevel.EMERGENCY: "Emergency condition - stop required"
        }
        return messages.get(level, "Unknown safety level")

    def _get_current_time(self) -> float:
        """Get current time"""
        import time
        return time.time()

    def get_safety_summary(self) -> Dict[str, Any]:
        """Get summary of safety status"""
        recent_events = [e for e in self.safety_events if e.timestamp > self._get_current_time() - 60.0]  # Last minute

        return {
            'current_level': self.current_level.name,
            'emergency_stop_active': self.emergency_stop_active,
            'safe_to_operate': self.is_safe_to_operate(),
            'total_events': len(self.safety_events),
            'recent_events': len(recent_events),
            'events_by_level': {
                level.name: len([e for e in recent_events if e.level == level])
                for level in SafetyLevel
            },
            'state_history_length': len(self.state_history)
        }

    def print_safety_status(self):
        """Print current safety status"""
        summary = self.get_safety_summary()

        print(f"\n🛡️  Safety Status:")
        print(f"   Level: {summary['current_level']}")
        print(f"   Safe to operate: {'✅' if summary['safe_to_operate'] else '❌'}")
        print(f"   Emergency stop: {'🚨 Active' if summary['emergency_stop_active'] else '✅ Clear'}")
        print(f"   Recent events (1min): {summary['recent_events']}")

        if summary['recent_events'] > 0:
            print("   Events by level:")
            for level, count in summary['events_by_level'].items():
                if count > 0:
                    print(f"     {level}: {count}")


def test_safety_monitor():
    """Test safety monitor functionality"""
    print("🧪 Testing Safety Monitor")

    # Mock UR10e specifications
    ur10e_specs = {
        'joint_limits': np.array([
            [-2*np.pi, 2*np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi],
            [-2*np.pi, 2*np.pi]
        ]),
        'torque_limits': np.array([330.0, 330.0, 330.0, 54.0, 54.0, 54.0])
    }

    # Create safety monitor
    safety = SafetyMonitor(ur10e_specs)

    # Test normal operation
    joint_pos = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
    joint_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joint_torque = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    tcp_pos = np.array([0.5, 0.3, 0.4])

    level = safety.check_safety(joint_pos, joint_vel, joint_torque, tcp_pos)
    print(f"✅ Normal operation: {level.name}")

    # Test torque limit warning
    high_torques = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 50.0])
    level = safety.check_safety(joint_pos, joint_vel, high_torques, tcp_pos)
    print(f"⚠️  High torques: {level.name}")

    # Test emergency condition
    dangerous_torques = np.array([400.0, 400.0, 400.0, 100.0, 100.0, 100.0])
    level = safety.check_safety(joint_pos, joint_vel, dangerous_torques, tcp_pos)
    print(f"🚨 Dangerous torques: {level.name}")

    # Test safe torque limiting
    safe_torques = safety.get_safe_torques(dangerous_torques)
    print(f"🛡️  Torque limiting: {dangerous_torques} → {safe_torques}")

    # Print safety status
    safety.print_safety_status()

    print("✅ Safety monitor test completed")


if __name__ == "__main__":
    test_safety_monitor()