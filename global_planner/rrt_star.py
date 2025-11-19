"""
RRT* (Rapidly-exploring Random Tree Star) Path Planner

Implementation of RRT* algorithm for UR10e robotic arm path planning.
Provides collision-free paths in joint space configuration.
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Node:
    """RRT* tree node"""
    position: np.ndarray    # 6D joint configuration
    cost: float             # Cost from root
    parent: Optional['Node'] = None
    children: List['Node'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class Waypoint:
    """Waypoint for RL local planner"""
    joint_positions: np.ndarray   # 6D joint angles
    cartesian_position: np.ndarray  # 3D end-effector position
    tolerance: float              # Position tolerance


class RRTStarPlanner:
    """
    RRT* path planner for UR10e robotic arm

    Features:
    - 6D joint space planning
    - Collision avoidance
    - Cost optimization (path length)
    - Dynamic constraints
    """

    def __init__(self,
                 joint_limits: np.ndarray,
                 max_iterations: int = 5000,
                 step_size: float = 0.1,
                 goal_bias: float = 0.1,
                 goal_tolerance: float = 0.1,
                 rewire_radius: float = None,
                 collision_check_resolution: int = 10):
        """
        Initialize RRT* planner

        Args:
            joint_limits: 6x2 array of joint limits [[min, max], ...]
            max_iterations: Maximum planning iterations
            step_size: Maximum step size for tree expansion
            goal_bias: Probability of sampling goal (0-1)
            goal_tolerance: Distance tolerance to goal
            rewire_radius: Radius for rewiring tree
            collision_check_resolution: Resolution for collision checking
        """
        self.joint_limits = joint_limits
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.collision_check_resolution = collision_check_resolution

        # Calculate rewire radius if not provided
        if rewire_radius is None:
            self.rewire_radius = 2.0 * step_size * math.sqrt(math.log(max_iterations) / max_iterations)
        else:
            self.rewire_radius = rewire_radius

        # Tree nodes
        self.nodes: List[Node] = []

        # Statistics
        self.stats = {
            'iterations': 0,
            'nodes_explored': 0,
            'path_length': 0.0,
            'planning_time': 0.0,
            'success': False
        }

    def plan(self,
             start: np.ndarray,
             goal: np.ndarray,
             obstacles: List[Dict] = None) -> Optional[List[Waypoint]]:
        """
        Plan path from start to goal configuration

        Args:
            start: 6D start joint configuration
            goal: 6D goal joint configuration
            obstacles: List of obstacles (simplified representation)

        Returns:
            List of waypoints if successful, None otherwise
        """
        import time
        start_time = time.time()

        print(f"🛤️  Starting RRT* planning...")
        print(f"   Start: {start}")
        print(f"   Goal: {goal}")
        print(f"   Max iterations: {self.max_iterations}")
        print(f"   Step size: {self.step_size}")

        # Validate inputs
        if not self._is_valid_configuration(start):
            print("❌ Start configuration is invalid")
            return None

        if not self._is_valid_configuration(goal):
            print("❌ Goal configuration is invalid")
            return None

        # Initialize tree with start node
        self.nodes = [Node(position=start.copy(), cost=0.0)]
        goal_node = None

        # RRT* main loop
        for iteration in range(self.max_iterations):
            self.stats['iterations'] = iteration + 1

            # Sample random configuration
            if np.random.random() < self.goal_bias:
                sample = goal.copy()
            else:
                sample = self._sample_random_configuration()

            # Find nearest node in tree
            nearest_node = self._find_nearest_node(sample)

            # Steer towards sample
            new_position = self._steer(nearest_node.position, sample)

            # Check if new position is valid
            if not self._is_valid_configuration(new_position):
                continue

            # Check collision-free path
            if not self._is_collision_free(nearest_node.position, new_position, obstacles):
                continue

            # Find best parent (cost optimization)
            best_parent = self._find_best_parent(new_position, nearest_node)

            # Create new node
            new_node = Node(
                position=new_position,
                cost=best_parent.cost + self._distance(best_parent.position, new_position),
                parent=best_parent
            )
            best_parent.children.append(new_node)
            self.nodes.append(new_node)

            # Rewire tree
            self._rewire_tree(new_node)

            # Check if goal is reached
            if self._distance(new_position, goal) < self.goal_tolerance:
                goal_node = new_node
                print(f"✅ Goal reached at iteration {iteration + 1}")
                break

            # Progress update
            if (iteration + 1) % 500 == 0:
                print(f"   Iteration {iteration + 1}: {len(self.nodes)} nodes explored")

        # Extract path if goal was reached
        if goal_node is not None:
            path = self._extract_path(goal_node)
            waypoints = self._create_waypoints(path)

            # Update statistics
            self.stats['planning_time'] = time.time() - start_time
            self.stats['nodes_explored'] = len(self.nodes)
            self.stats['path_length'] = sum(self._distance(waypoints[i].joint_positions,
                                                         waypoints[i+1].joint_positions)
                                           for i in range(len(waypoints)-1))
            self.stats['success'] = True

            print(f"🎯 Planning successful!")
            print(f"   Path waypoints: {len(waypoints)}")
            print(f"   Path length: {self.stats['path_length']:.4f}")
            print(f"   Planning time: {self.stats['planning_time']:.3f}s")

            return waypoints
        else:
            self.stats['planning_time'] = time.time() - start_time
            self.stats['success'] = False
            print(f"❌ Planning failed: goal not reached after {self.max_iterations} iterations")
            return None

    def _sample_random_configuration(self) -> np.ndarray:
        """Sample random valid configuration"""
        config = np.zeros(6)
        for i in range(6):
            config[i] = np.random.uniform(self.joint_limits[i, 0], self.joint_limits[i, 1])
        return config

    def _find_nearest_node(self, position: np.ndarray) -> Node:
        """Find nearest node in tree to given position"""
        min_distance = float('inf')
        nearest_node = self.nodes[0]

        for node in self.nodes:
            distance = self._distance(node.position, position)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from one position towards another"""
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_pos
        else:
            return from_pos + (direction / distance) * self.step_size

    def _distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two configurations"""
        # Weighted distance (can be customized per joint)
        weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])  # Lower weight for wrist joints
        return np.linalg.norm(weights * (pos1 - pos2))

    def _is_valid_configuration(self, config: np.ndarray) -> bool:
        """Check if configuration is within joint limits"""
        for i in range(6):
            if config[i] < self.joint_limits[i, 0] - 0.01 or config[i] > self.joint_limits[i, 1] + 0.01:
                return False
        return True

    def _is_collision_free(self, from_pos: np.ndarray, to_pos: np.ndarray, obstacles: List[Dict] = None) -> bool:
        """
        Check if path between two positions is collision-free

        Simplified collision checking - can be extended with proper collision detection
        """
        # Simple collision check using intermediate configurations
        steps = self.collision_check_resolution
        for i in range(steps + 1):
            t = i / steps
            intermediate = from_pos + t * (to_pos - from_pos)

            if not self._is_valid_configuration(intermediate):
                return False

        # TODO: Add proper collision detection with obstacles
        if obstacles:
            for obstacle in obstacles:
                if self._check_obstacle_collision(intermediate, obstacle):
                    return False

        return True

    def _check_obstacle_collision(self, config: np.ndarray, obstacle: Dict) -> bool:
        """Check if configuration collides with obstacle"""
        # Simplified obstacle checking - replace with proper geometric collision detection
        return False

    def _find_best_parent(self, position: np.ndarray, default_parent: Node) -> Node:
        """Find best parent for new position (cost optimization)"""
        # Find nodes within rewire radius
        nearby_nodes = []
        for node in self.nodes:
            if self._distance(node.position, position) <= self.rewire_radius:
                if self._is_collision_free(node.position, position):
                    nearby_nodes.append(node)

        if not nearby_nodes:
            return default_parent

        # Find node with minimum cost
        best_parent = nearby_nodes[0]
        min_cost = nearby_nodes[0].cost + self._distance(nearby_nodes[0].position, position)

        for node in nearby_nodes[1:]:
            cost = node.cost + self._distance(node.position, position)
            if cost < min_cost:
                min_cost = cost
                best_parent = node

        return best_parent

    def _rewire_tree(self, new_node: Node):
        """Rewire tree for cost optimization"""
        # Find nodes within rewire radius
        for node in self.nodes:
            if node == new_node:
                continue

            if self._distance(node.position, new_node.position) <= self.rewire_radius:
                if self._is_collision_free(new_node.position, node.position):
                    new_cost = new_node.cost + self._distance(new_node.position, node.position)

                    if new_cost < node.cost:
                        # Rewire
                        if node.parent:
                            node.parent.children.remove(node)

                        node.parent = new_node
                        node.cost = new_cost
                        new_node.children.append(node)

    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """Extract path from root to goal node"""
        path = []
        current = goal_node

        while current is not None:
            path.append(current.position.copy())
            current = current.parent

        path.reverse()
        return path

    def _create_waypoints(self, path: List[np.ndarray]) -> List[Waypoint]:
        """Create waypoints from path"""
        waypoints = []

        for i, config in enumerate(path):
            # Convert joint configuration to Cartesian position (simplified)
            cartesian_pos = self._forward_kinematics(config)

            waypoint = Waypoint(
                joint_positions=config,
                cartesian_position=cartesian_pos,
                tolerance=0.05 if i < len(path) - 1 else 0.01  # Tighter tolerance for goal
            )
            waypoints.append(waypoint)

        return waypoints

    def _forward_kinematics(self, joint_config: np.ndarray) -> np.ndarray:
        """
        Simplified forward kinematics for UR10e
        Returns approximate end-effector position
        """
        # Simplified UR10e kinematics
        a1, a2, a3 = 0.612, 0.572, 0.174
        d1, d4, d6 = 0.1273, 0.1199, 0.11655

        q1, q2, q3, q4, q5, q6 = joint_config

        x = (a1 * np.cos(q1) + a2 * np.cos(q1) * np.cos(q2) +
             a3 * np.cos(q1) * np.cos(q2 + q3))
        y = (a1 * np.sin(q1) + a2 * np.sin(q1) * np.cos(q2) +
             a3 * np.sin(q1) * np.cos(q2 + q3))
        z = d1 + a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + d4

        return np.array([x, y, z])

    def visualize_tree(self, save_path: str = None):
        """Visualize RRT* tree (2D projection)"""
        if not self.nodes:
            print("⚠️ No tree to visualize")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot tree structure
        positions = np.array([node.position for node in self.nodes])

        # Plot first two joints
        ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=5, alpha=0.6)
        ax1.set_xlabel('Joint 1 (rad)')
        ax1.set_ylabel('Joint 2 (rad)')
        ax1.set_title('RRT* Tree (Joints 1-2)')
        ax1.grid(True)

        # Plot edges
        for node in self.nodes:
            if node.parent:
                ax1.plot([node.parent.position[0], node.position[0]],
                        [node.parent.position[1], node.position[1]], 'b-', alpha=0.3)

        # Plot joints 2-3
        ax2.scatter(positions[:, 1], positions[:, 2], c='blue', s=5, alpha=0.6)
        ax2.set_xlabel('Joint 2 (rad)')
        ax2.set_ylabel('Joint 3 (rad)')
        ax2.set_title('RRT* Tree (Joints 2-3)')
        ax2.grid(True)

        # Plot edges
        for node in self.nodes:
            if node.parent:
                ax2.plot([node.parent.position[1], node.position[1]],
                        [node.parent.position[2], node.position[2]], 'b-', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Tree visualization saved to {save_path}")
        else:
            plt.show()

    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        return self.stats.copy()


def test_rrt_star():
    """Test RRT* planner with example"""
    print("🧪 Testing RRT* Planner")

    # Define UR10e joint limits
    joint_limits = np.array([
        [-2*np.pi, 2*np.pi],      # Shoulder pan
        [-np.pi, np.pi],          # Shoulder lift
        [-np.pi, np.pi],          # Elbow
        [-2*np.pi, 2*np.pi],      # Wrist 1
        [-2*np.pi, 2*np.pi],      # Wrist 2
        [-2*np.pi, 2*np.pi]       # Wrist 3
    ])

    # Create planner
    planner = RRTStarPlanner(
        joint_limits=joint_limits,
        max_iterations=2000,
        step_size=0.2,
        goal_bias=0.1
    )

    # Define start and goal
    start = np.array([0.0, -np.pi/2, np.pi/2, 0.0, np.pi/2, 0.0])
    goal = np.array([np.pi/4, -np.pi/3, np.pi/3, np.pi/6, np.pi/4, -np.pi/6])

    # Plan path
    waypoints = planner.plan(start, goal)

    if waypoints:
        print(f"✅ Planning successful with {len(waypoints)} waypoints")

        # Print first few waypoints
        for i, waypoint in enumerate(waypoints[:3]):
            print(f"   Waypoint {i+1}: Joints={waypoint.joint_positions}, Pos={waypoint.cartesian_position}")

        # Print statistics
        stats = planner.get_statistics()
        print(f"📊 Planning Statistics: {stats}")

        # Visualize tree (optional)
        # planner.visualize_tree()
    else:
        print("❌ Planning failed")


if __name__ == "__main__":
    test_rrt_star()