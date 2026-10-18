"""Behaviour tests for planning, geometry and closed-loop navigation."""

import math
import unittest

from navigation import (GridMap, NoPathError, Pose, RobotConfig, demo_map,
                        drive_step, metrics, motion_is_free, plan_astar, simulate,
                        wheel_speeds)


class PlanningTests(unittest.TestCase):
    def test_shortest_path_in_empty_map(self):
        grid = GridMap(8, 8, 0.5, frozenset())
        path = plan_astar(grid, (1, 1), (6, 4))
        self.assertEqual(path[0], (1, 1))
        self.assertEqual(path[-1], (6, 4))
        self.assertEqual(len(path) - 1, 8)  # independent Manhattan lower bound

    def test_unreachable_goal_reports_failure(self):
        wall = frozenset((3, y) for y in range(7))
        grid = GridMap(7, 7, 0.5, wall)
        with self.assertRaises(NoPathError):
            plan_astar(grid, (1, 3), (5, 3))

    def test_planned_path_avoids_obstacles_with_robot_clearance(self):
        grid, start, goal = demo_map()
        path = plan_astar(grid, start, goal)
        for first, second in zip(path, path[1:]):
            self.assertEqual(abs(first[0] - second[0]) + abs(first[1] - second[1]), 1)
            self.assertTrue(grid.segment_is_free(grid.centre(first), grid.centre(second), 0.18))

    def test_occupied_goal_is_invalid(self):
        grid = GridMap(5, 5, 0.5, frozenset({(3, 3)}))
        with self.assertRaises(ValueError):
            plan_astar(grid, (1, 1), (3, 3))

    def test_same_start_and_goal(self):
        grid = GridMap(4, 4, 0.5, frozenset())
        path = plan_astar(grid, (2, 2), (2, 2))
        result = simulate(grid, path)
        self.assertEqual(path, [(2, 2)])
        self.assertTrue(result.reached_goal)
        self.assertEqual(len(result.wheel_commands), 0)


class GeometryTests(unittest.TestCase):
    def test_footprint_detects_obstacle_corner(self):
        grid = GridMap(5, 5, 1.0, frozenset({(2, 2)}))
        self.assertTrue(grid.is_free((1.8, 1.8), 0.2))
        self.assertFalse(grid.is_free((1.8, 1.8), 0.3))

    def test_footprint_cannot_leave_map(self):
        grid = GridMap(5, 5, 1.0, frozenset())
        self.assertFalse(grid.is_free((0.1, 2.0), 0.2))
        self.assertTrue(grid.is_free((0.3, 2.0), 0.2))

    def test_collision_guard_checks_between_safe_endpoints(self):
        grid = GridMap(8, 5, 0.5, frozenset({(3, 2)}))
        config = RobotConfig(dt=3.0)
        start = Pose(1.25, 1.25, 0.0)
        left, right = wheel_speeds(0.5, 0.0, config)
        end = drive_step(start, left, right, config.dt, config)
        self.assertTrue(grid.is_free(start.point, config.radius))
        self.assertTrue(grid.is_free(end.point, config.radius))
        self.assertFalse(motion_is_free(grid, start, left, right, config))


class KinematicsTests(unittest.TestCase):
    def test_equal_wheels_drive_straight(self):
        config = RobotConfig()
        end = drive_step(Pose(1.0, 1.0, 0.0), 10.0, 10.0, 2.0, config)
        self.assertAlmostEqual(end.x, 2.0)
        self.assertAlmostEqual(end.y, 1.0)
        self.assertAlmostEqual(end.heading, 0.0)

    def test_opposite_wheels_rotate_without_translation(self):
        config = RobotConfig()
        end = drive_step(Pose(1.0, 1.0, 0.0), -5.0, 5.0, 0.5, config)
        self.assertAlmostEqual(end.x, 1.0)
        self.assertAlmostEqual(end.y, 1.0)
        self.assertAlmostEqual(end.heading, 0.05 * 10 / 0.28 * 0.5)

    def test_wheel_speed_limits_preserve_ratio(self):
        config = RobotConfig()
        left, right = wheel_speeds(3.0, 4.0, config)
        self.assertLessEqual(max(abs(left), abs(right)), config.max_wheel_speed)
        self.assertAlmostEqual(left / right, (3.0 - 4.0 * 0.14) / (3.0 + 4.0 * 0.14))


class ControlTests(unittest.TestCase):
    def test_controller_reaches_demo_goal_without_collision(self):
        grid, start, goal = demo_map()
        config = RobotConfig()
        path = plan_astar(grid, start, goal, config.radius)
        result = simulate(grid, path, config, initial_heading=math.pi)
        self.assertTrue(result.reached_goal)
        self.assertLessEqual(math.dist(result.poses[-1].point, grid.centre(goal)), config.goal_tolerance)
        self.assertTrue(all(grid.is_free(pose.point, config.radius) for pose in result.poses))
        self.assertTrue(all(motion_is_free(grid, pose, *command, config)
                            for pose, command in zip(result.poses, result.wheel_commands)))
        self.assertLess(len(result.wheel_commands), 2000)

    def test_timeout_is_reported_as_failure(self):
        grid, start, goal = demo_map()
        result = simulate(grid, plan_astar(grid, start, goal), max_steps=1)
        self.assertFalse(result.reached_goal)

    def test_invalid_shortcut_cannot_pass_through_wall(self):
        grid = GridMap(8, 5, 0.5, frozenset({(3, 2)}))
        with self.assertRaises(ValueError):
            simulate(grid, [(1, 2), (6, 2)])

    def test_demo_repeats_exactly(self):
        grid, start, goal = demo_map()
        config = RobotConfig()
        path = plan_astar(grid, start, goal)
        first = metrics(grid, path, simulate(grid, path), config)
        second = metrics(grid, path, simulate(grid, path), config)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
