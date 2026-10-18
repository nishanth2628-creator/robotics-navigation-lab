"""Small, inspectable 2D navigation simulation. All distances are in metres."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math

Cell = tuple[int, int]
Point = tuple[float, float]


class NoPathError(RuntimeError):
    """The valid start and goal are disconnected in the occupancy grid."""


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    resolution: float
    occupied: frozenset[Cell]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.resolution <= 0:
            raise ValueError("Map dimensions and resolution must be positive.")
        if any(not self.contains(cell) for cell in self.occupied):
            raise ValueError("An occupied cell is outside the map.")

    def contains(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def centre(self, cell: Cell) -> Point:
        if not self.contains(cell):
            raise ValueError(f"Cell {cell} is outside the map.")
        x, y = cell
        return ((x + 0.5) * self.resolution, (y + 0.5) * self.resolution)

    def is_free(self, point: Point, radius: float = 0.0) -> bool:
        """Check a circular footprint against square obstacles and map edges.

        A footprint touching an obstacle is treated as a collision. Unlike a
        cell-only check, this also checks clearance from the obstacle corners.
        """
        x, y = point
        if not all(math.isfinite(value) for value in (x, y, radius)) or radius < 0:
            raise ValueError("Coordinates must be finite and radius non-negative.")
        size = self.resolution
        if x - radius <= 0 or y - radius <= 0:
            return False
        if x + radius >= self.width * size or y + radius >= self.height * size:
            return False

        # Only cells overlapping the bounding box of the circular footprint
        # can collide with it. Include the cell immediately before an exact
        # boundary so obstacle contact is handled consistently.
        min_x = max(0, math.floor((x - radius) / size) - 1)
        max_x = min(self.width - 1, math.floor((x + radius) / size))
        min_y = max(0, math.floor((y - radius) / size) - 1)
        max_y = min(self.height - 1, math.floor((y + radius) / size))
        for cx in range(min_x, max_x + 1):
            for cy in range(min_y, max_y + 1):
                if (cx, cy) not in self.occupied:
                    continue
                nearest_x = min(max(x, cx * size), (cx + 1) * size)
                nearest_y = min(max(y, cy * size), (cy + 1) * size)
                if (x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius**2:
                    return False
        return True

    def segment_is_free(self, start: Point, end: Point, radius: float) -> bool:
        """Sample a straight segment at intervals no larger than 2 cm."""
        distance = math.dist(start, end)
        spacing = min(0.02, self.resolution / 10.0)
        steps = max(1, math.ceil(distance / spacing))
        return all(
            self.is_free(
                (start[0] + (end[0] - start[0]) * step / steps,
                 start[1] + (end[1] - start[1]) * step / steps),
                radius,
            )
            for step in range(steps + 1)
        )


def plan_astar(grid: GridMap, start: Cell, goal: Cell, robot_radius: float = 0.18) -> list[Cell]:
    """Find a shortest four-connected path for a circular robot footprint.

    Manhattan distance is an admissible heuristic here: every move is one
    horizontal or vertical cell. Obstacles are checked at the footprint size.
    """
    for label, cell in (("start", start), ("goal", goal)):
        if not grid.contains(cell) or not grid.is_free(grid.centre(cell), robot_radius):
            raise ValueError(f"The {label} cell {cell} cannot contain the robot.")

    def heuristic(cell: Cell) -> float:
        return (abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])) * grid.resolution

    counter = itertools.count()
    frontier = [(heuristic(start), next(counter), start)]
    costs = {start: 0.0}
    parents: dict[Cell, Cell] = {}
    visited: set[Cell] = set()
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            return list(reversed(path))
        visited.add(current)
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            neighbour = (current[0] + dx, current[1] + dy)
            if not grid.contains(neighbour) or not grid.is_free(grid.centre(neighbour), robot_radius):
                continue
            if not grid.segment_is_free(grid.centre(current), grid.centre(neighbour), robot_radius):
                continue
            new_cost = costs[current] + grid.resolution
            if new_cost < costs.get(neighbour, math.inf):
                costs[neighbour] = new_cost
                parents[neighbour] = current
                heapq.heappush(frontier, (new_cost + heuristic(neighbour), next(counter), neighbour))
    raise NoPathError("No collision-free path connects the start and goal.")


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float  # radians, anticlockwise from the positive x-axis

    @property
    def point(self) -> Point:
        return (self.x, self.y)


@dataclass(frozen=True)
class RobotConfig:
    radius: float = 0.18
    wheel_radius: float = 0.05
    wheel_base: float = 0.28
    max_wheel_speed: float = 16.0  # rad/s
    max_linear_speed: float = 0.55  # m/s
    max_angular_speed: float = 1.8  # rad/s
    heading_gain: float = 2.5
    distance_gain: float = 1.6
    waypoint_tolerance: float = 0.10
    goal_tolerance: float = 0.10
    dt: float = 0.10

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in self.__dict__.values()):
            raise ValueError("Robot configuration values must be finite and positive.")


def wheel_speeds(linear: float, angular: float, config: RobotConfig) -> tuple[float, float]:
    """Convert body velocities to wheel rates, preserving curvature if clipped."""
    left = (linear - angular * config.wheel_base / 2) / config.wheel_radius
    right = (linear + angular * config.wheel_base / 2) / config.wheel_radius
    scale = max(1.0, abs(left) / config.max_wheel_speed, abs(right) / config.max_wheel_speed)
    return left / scale, right / scale


def drive_step(pose: Pose, left: float, right: float, dt: float, config: RobotConfig) -> Pose:
    """Integrate ideal differential-drive kinematics exactly for fixed wheel rates."""
    if not all(math.isfinite(value) for value in (left, right, dt)) or dt < 0:
        raise ValueError("Wheel rates and time must be finite; time cannot be negative.")
    linear = config.wheel_radius * (left + right) / 2
    angular = config.wheel_radius * (right - left) / config.wheel_base
    new_heading = pose.heading + angular * dt
    if abs(angular) < 1e-10:
        x = pose.x + linear * math.cos(pose.heading) * dt
        y = pose.y + linear * math.sin(pose.heading) * dt
    else:
        x = pose.x + (linear / angular) * (math.sin(new_heading) - math.sin(pose.heading))
        y = pose.y - (linear / angular) * (math.cos(new_heading) - math.cos(pose.heading))
    return Pose(x, y, wrap_angle(new_heading))


def motion_is_free(grid: GridMap, pose: Pose, left: float, right: float, config: RobotConfig) -> bool:
    """Check the proposed curved motion at <=2 cm and <=0.04 rad intervals."""
    linear = config.wheel_radius * (left + right) / 2
    angular = config.wheel_radius * (right - left) / config.wheel_base
    spacing = min(0.02, grid.resolution / 10.0)
    steps = max(1, math.ceil(abs(linear) * config.dt / spacing),
                math.ceil(abs(angular) * config.dt / 0.04))
    return all(
        grid.is_free(drive_step(pose, left, right, config.dt * step / steps, config).point, config.radius)
        for step in range(steps + 1)
    )


@dataclass
class SimulationResult:
    poses: list[Pose]
    wheel_commands: list[tuple[float, float]]
    reached_goal: bool
    rejected_commands: int
    dt: float


def simulate(grid: GridMap, path: list[Cell], config: RobotConfig | None = None,
             initial_heading: float = 0.0, max_steps: int = 5000) -> SimulationResult:
    """Track successive waypoints with heading feedback and a collision guard.

    If a candidate motion collides, stop its translation and rotate toward the
    next waypoint instead. This is a safety fallback, not a replanning method.
    """
    config = config or RobotConfig()
    if not path:
        raise ValueError("A non-empty path is required.")
    if max_steps < 0 or not math.isfinite(initial_heading):
        raise ValueError("Step count must be non-negative and heading finite.")
    points = [grid.centre(cell) for cell in path]
    if any(not grid.is_free(point, config.radius) for point in points):
        raise ValueError("A path waypoint collides with the robot footprint.")
    if any(not grid.segment_is_free(a, b, config.radius) for a, b in zip(points, points[1:])):
        raise ValueError("A path segment crosses an obstacle.")

    pose = Pose(*points[0], wrap_angle(initial_heading))
    poses = [pose]
    commands: list[tuple[float, float]] = []
    target_index = min(1, len(points) - 1)
    rejected = 0
    reached = math.dist(pose.point, points[-1]) <= config.goal_tolerance
    for _ in range(max_steps):
        if reached:
            break
        target = points[target_index]
        distance = math.dist(pose.point, target)
        if target_index < len(points) - 1 and distance <= config.waypoint_tolerance:
            target_index += 1
            target = points[target_index]
            distance = math.dist(pose.point, target)

        error = wrap_angle(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.heading)
        angular = max(-config.max_angular_speed, min(config.max_angular_speed, config.heading_gain * error))
        # Turn in place when the target is well off-axis; otherwise slow on
        # approach and reduce speed smoothly as heading error grows.
        linear = min(config.max_linear_speed, config.distance_gain * distance) * max(0.0, math.cos(error))
        if abs(error) > 0.8:
            linear = 0.0
        left, right = wheel_speeds(linear, angular, config)
        if not motion_is_free(grid, pose, left, right, config):
            rejected += 1
            left, right = wheel_speeds(0.0, angular, config)
            if not motion_is_free(grid, pose, left, right, config):
                break
        pose = drive_step(pose, left, right, config.dt, config)
        poses.append(pose)
        commands.append((left, right))
        reached = math.dist(pose.point, points[-1]) <= config.goal_tolerance

    return SimulationResult(poses, commands, reached, rejected, config.dt)


def demo_map() -> tuple[GridMap, Cell, Cell]:
    """Deterministic warehouse-like map; synthetic, not measured hardware data."""
    width, height = 24, 18
    occupied = {(x, y) for x in range(width) for y in range(height)
                if x in (0, width - 1) or y in (0, height - 1)}
    occupied.update((7, y) for y in range(1, 14) if y not in (5, 6))
    occupied.update((x, 10) for x in range(7, 23) if x not in (15, 16))
    occupied.update((x, y) for x in range(11, 14) for y in range(3, 7))
    occupied.update((x, y) for x in range(18, 21) for y in range(12, 15))
    return GridMap(width, height, 0.5, frozenset(occupied)), (2, 2), (21, 15)


def path_length(points: list[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def metrics(grid: GridMap, path: list[Cell], result: SimulationResult, config: RobotConfig) -> dict:
    """Measured deterministic simulation results, not robot performance claims."""
    return {
        "scenario": "synthetic_warehouse_v1",
        "reached_goal": result.reached_goal,
        "planned_path_length_m": round(path_length([grid.centre(cell) for cell in path]), 6),
        "sampled_trajectory_length_m": round(path_length([pose.point for pose in result.poses]), 6),
        "trajectory_length_method": "Sum of straight-line distances between recorded poses; not exact arc length.",
        "final_goal_error_m": round(math.dist(result.poses[-1].point, grid.centre(path[-1])), 6),
        "simulated_time_s": round(len(result.wheel_commands) * result.dt, 6),
        "control_steps": len(result.wheel_commands),
        "collision_free_recorded_poses": all(grid.is_free(pose.point, config.radius) for pose in result.poses),
        "collision_free_motion_samples": all(
            motion_is_free(grid, pose, *command, config)
            for pose, command in zip(result.poses, result.wheel_commands)
        ),
        "rejected_candidate_commands": result.rejected_commands,
        "peak_wheel_speed_rad_s": round(max((abs(value) for command in result.wheel_commands for value in command), default=0), 6),
        "robot_radius_m": config.radius,
        "goal_tolerance_m": config.goal_tolerance,
        "dt_s": config.dt,
        "obstacle_cell_count": len(grid.occupied),
        "grid_resolution_m": grid.resolution,
        "collision_check_max_translation_spacing_m": min(0.02, grid.resolution / 10.0),
        "collision_check_max_rotation_spacing_rad": 0.04,
    }
