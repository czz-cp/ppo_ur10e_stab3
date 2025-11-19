"""
Global Planner Interface

Integration interface between RRT* global planning and RL local control.
Provides waypoint management and coordination between planning layers.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time

from .rrt_star import RRTStarPlanner, Waypoint


@dataclass
class PlanningRequest:
    """Request for global path planning"""
    start_position: np.ndarray          # Current 6D joint configuration
    target_position: np.ndarray         # Target 3D Cartesian position
    obstacles: List[Dict] = None        # Known obstacles
    max_planning_time: float = 5.0      # Maximum planning time (seconds)
    tolerance: float = 0.05             # Goal tolerance (meters)


@dataclass
class PlanningResult:
    """Result of global path planning"""
    success: bool                       # Whether planning succeeded
    waypoints: List[Waypoint] = None    # Planned waypoints
    planning_time: float = 0.0         # Time taken for planning
    total_distance: float = 0.0        # Total path distance
    error_message: str = ""            # Error message if failed


class GlobalPlannerInterface:
    """
    Interface between global RRT* planning and local RL control

    Responsibilities:
    - Manage global planning requests
    - Generate waypoints for RL agent
    - Handle replanning when needed
    - Coordinate between planning and control layers
    """

    def __init__(self,
                 ur10e_joint_limits: np.ndarray,
                 waypoint_spacing: float = 0.1,
                 replanning_threshold: float = 0.1,
                 max_waypoints: int = 50):
        """
        Initialize global planner interface

        Args:
            ur10e_joint_limits: 6x2 array of joint limits
            waypoint_spacing: Desired spacing between waypoints (meters)
            replanning_threshold: Distance threshold for replanning (meters)
            max_waypoints: Maximum number of waypoints in plan
        """
        # Initialize RRT* planner
        self.rrt_planner = RRTStarPlanner(
            joint_limits=ur10e_joint_limits,
            max_iterations=5000,
            step_size=0.1,
            goal_bias=0.1,
            goal_tolerance=0.05
        )

        # Configuration
        self.waypoint_spacing = waypoint_spacing
        self.replanning_threshold = replanning_threshold
        self.max_waypoints = max_waypoints

        # Current plan
        self.current_waypoints: List[Waypoint] = []
        self.current_waypoint_index: int = 0
        self.last_planning_time: float = 0.0

        # Statistics
        self.stats = {
            'planning_requests': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'replanning_events': 0,
            'total_planning_time': 0.0,
            'average_planning_time': 0.0
        }

    def plan_to_target(self, request: PlanningRequest) -> PlanningResult:
        """
        Plan path to target position

        Args:
            request: Planning request with start, target, and constraints

        Returns:
            PlanningResult with waypoints or error information
        """
        self.stats['planning_requests'] += 1

        start_time = time.time()

        print(f"🛤️  Global planning request:")
        print(f"   Start: {request.start_position}")
        print(f"   Target: {request.target_position}")
        print(f"   Max time: {request.max_planning_time}s")

        # Convert target Cartesian position to joint configuration
        # This is a simplified inverse kinematics - in practice use proper IK solver
        target_joint_config = self._inverse_kinematics(request.target_position, request.start_position)

        if target_joint_config is None:
            return PlanningResult(
                success=False,
                error_message="Failed to compute inverse kinematics for target position"
            )

        # Plan path using RRT*
        waypoints = self.rrt_planner.plan(
            start=request.start_position,
            goal=target_joint_config,
            obstacles=request.obstacles
        )

        planning_time = time.time() - start_time

        if waypoints:
            # Update statistics
            self.stats['successful_plans'] += 1
            self.stats['total_planning_time'] += planning_time
            self.stats['average_planning_time'] = self.stats['total_planning_time'] / self.stats['successful_plans']

            # Limit waypoints
            if len(waypoints) > self.max_waypoints:
                waypoints = self._sample_waypoints(waypoints, self.max_waypoints)

            # Update current plan
            self.current_waypoints = waypoints
            self.current_waypoint_index = 0
            self.last_planning_time = planning_time

            # Calculate total distance
            total_distance = sum(
                np.linalg.norm(waypoints[i].joint_positions - waypoints[i+1].joint_positions)
                for i in range(len(waypoints)-1)
            )

            print(f"✅ Global planning successful:")
            print(f"   Waypoints: {len(waypoints)}")
            print(f"   Planning time: {planning_time:.3f}s")
            print(f"   Total distance: {total_distance:.4f}")

            return PlanningResult(
                success=True,
                waypoints=waypoints,
                planning_time=planning_time,
                total_distance=total_distance
            )
        else:
            # Update statistics
            self.stats['failed_plans'] += 1

            error_msg = "RRT* planning failed to find path"
            print(f"❌ Global planning failed: {error_msg}")

            return PlanningResult(
                success=False,
                planning_time=planning_time,
                error_message=error_msg
            )

    def get_current_waypoint(self) -> Optional[Waypoint]:
        """
        Get current waypoint for RL agent to target

        Returns:
            Current waypoint or None if no plan available
        """
        if not self.current_waypoints or self.current_waypoint_index >= len(self.current_waypoints):
            return None

        return self.current_waypoints[self.current_waypoint_index]

    def advance_to_next_waypoint(self) -> bool:
        """
        Advance to next waypoint in plan

        Returns:
            True if advanced successfully, False if no more waypoints
        """
        if self.current_waypoint_index < len(self.current_waypoints) - 1:
            self.current_waypoint_index += 1
            print(f"📍 Advanced to waypoint {self.current_waypoint_index + 1}/{len(self.current_waypoints)}")
            return True
        else:
            print("🎯 Reached final waypoint")
            return False

    def check_replanning_needed(self, current_position: np.ndarray, current_target: np.ndarray) -> bool:
        """
        Check if replanning is needed due to deviation from plan

        Args:
            current_position: Current joint configuration
            current_target: Current target position

        Returns:
            True if replanning is needed
        """
        if not self.current_waypoints:
            return True

        # Check if we've reached the final waypoint
        if self.current_waypoint_index >= len(self.current_waypoints) - 1:
            final_waypoint = self.current_waypoints[-1]
            distance_to_goal = np.linalg.norm(current_position - final_waypoint.joint_positions)
            if distance_to_goal < final_waypoint.tolerance:
                return False  # Goal reached

        # Check if target has changed significantly
        current_waypoint = self.get_current_waypoint()
        if current_waypoint:
            target_distance = np.linalg.norm(current_target - current_waypoint.cartesian_position)
            if target_distance > self.replanning_threshold:
                print(f"🔄 Target deviation: {target_distance:.4f}m > {self.replanning_threshold:.4f}m, replanning needed")
                return True

        # Check if we're too far from current waypoint
        if current_waypoint:
            waypoint_distance = np.linalg.norm(current_position - current_waypoint.joint_positions)
            if waypoint_distance > current_waypoint.tolerance * 2:  # Allow some tolerance
                print(f"🔄 Waypoint deviation: {waypoint_distance:.4f}rad, replanning may be needed")
                return True

        return False

    def get_local_target_for_rl(self) -> Optional[np.ndarray]:
        """
        Get target position for RL local controller

        Returns:
            6D joint target or None if no waypoint available
        """
        waypoint = self.get_current_waypoint()
        if waypoint:
            return waypoint.joint_positions
        return None

    def get_remaining_waypoints(self) -> int:
        """Get number of remaining waypoints"""
        if not self.current_waypoints:
            return 0
        return len(self.current_waypoints) - self.current_waypoint_index

    def reset_plan(self):
        """Reset current plan"""
        self.current_waypoints = []
        self.current_waypoint_index = 0
        print("🔄 Global plan reset")

    def get_planning_progress(self) -> Dict[str, Any]:
        """
        Get current planning progress

        Returns:
            Dictionary with progress information
        """
        if not self.current_waypoints:
            return {
                'has_plan': False,
                'current_waypoint': None,
                'total_waypoints': 0,
                'remaining_waypoints': 0,
                'progress_percentage': 0.0
            }

        progress = (self.current_waypoint_index / len(self.current_waypoints)) * 100

        return {
            'has_plan': True,
            'current_waypoint': self.current_waypoint_index + 1,
            'total_waypoints': len(self.current_waypoints),
            'remaining_waypoints': self.get_remaining_waypoints(),
            'progress_percentage': progress
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        stats = self.stats.copy()
        stats.update(self.rrt_planner.get_statistics())
        return stats

    def _inverse_kinematics(self, target_position: np.ndarray, reference_config: np.ndarray) -> Optional[np.ndarray]:
        """
        Simplified inverse kinematics for UR10e

        Args:
            target_position: 3D target position
            reference_config: Reference joint configuration

        Returns:
            6D joint configuration or None if no solution
        """
        # This is a simplified IK solver - in practice use a proper IK solver
        # For now, we'll use a heuristic approach

        x, y, z = target_position

        # Check if target is reachable
        max_reach = 0.612 + 0.572 + 0.174  # Sum of link lengths
        distance = np.sqrt(x**2 + y**2 + (z - 0.1273)**2)

        if distance > max_reach * 0.95:  # Leave some margin
            print(f"⚠️ Target position {target_position} may be unreachable (distance: {distance:.3f}m)")

        # Simplified IK (very approximate)
        q1 = np.arctan2(y, x)
        q2 = -np.pi/2 + np.arctan2(z - 0.1273, np.sqrt(x**2 + y**2))
        q3 = np.pi/2 - q2
        q4 = reference_config[3]  # Keep wrist orientation from reference
        q5 = reference_config[4]
        q6 = reference_config[5]

        joint_config = np.array([q1, q2, q3, q4, q5, q6])

        # Validate joint limits
        # TODO: Add proper joint limit checking

        return joint_config

    def _sample_waypoints(self, waypoints: List[Waypoint], target_count: int) -> List[Waypoint]:
        """
        Sample waypoints to reduce count while preserving path shape

        Args:
            waypoints: Original waypoints
            target_count: Target number of waypoints

        Returns:
            Sampled waypoints
        """
        if len(waypoints) <= target_count:
            return waypoints

        # Always include first and last waypoints
        sampled = [waypoints[0]]

        # Sample intermediate waypoints
        step = (len(waypoints) - 1) / (target_count - 1)
        for i in range(1, target_count - 1):
            index = int(i * step)
            sampled.append(waypoints[index])

        sampled.append(waypoints[-1])

        return sampled


def test_global_planner_interface():
    """Test global planner interface"""
    print("🧪 Testing Global Planner Interface")

    # Define UR10e joint limits
    joint_limits = np.array([
        [-2*np.pi, 2*np.pi],
        [-np.pi, np.pi],
        [-np.pi, np.pi],
        [-2*np.pi, 2*np.pi],
        [-2*np.pi, 2*np.pi],
        [-2*np.pi, 2*np.pi]
    ])

    # Create interface
    interface = GlobalPlannerInterface(
        ur10e_joint_limits=joint_limits,
        waypoint_spacing=0.1,
        replanning_threshold=0.1
    )

    # Create planning request
    start_config = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
    target_position = np.array([0.5, 0.3, 0.4])

    request = PlanningRequest(
        start_position=start_config,
        target_position=target_position,
        max_planning_time=5.0
    )

    # Plan path
    result = interface.plan_to_target(request)

    if result.success:
        print(f"✅ Planning successful with {len(result.waypoints)} waypoints")

        # Test waypoint progression
        waypoint = interface.get_current_waypoint()
        print(f"   Current waypoint: {waypoint.cartesian_position if waypoint else None}")

        # Test statistics
        stats = interface.get_statistics()
        print(f"📊 Statistics: {stats}")

        # Test progress
        progress = interface.get_planning_progress()
        print(f"📍 Progress: {progress}")
    else:
        print(f"❌ Planning failed: {result.error_message}")


if __name__ == "__main__":
    test_global_planner_interface()