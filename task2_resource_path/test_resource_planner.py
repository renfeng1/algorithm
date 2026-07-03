import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from common.models import MazeGame
from task2_resource_path.resource_planner import plan_optimal_resource_path

class TestResourcePlanner(unittest.TestCase):
    def _create_maze(self, grid_str_list):
        grid = [list(row) for row in grid_str_list]
        return MazeGame(
            grid=grid,
            boss_health=[],
            player_skills=[],
            min_rounds=0,
            coin_consumption=0
        )

    def test_straight_path(self):
        grid = [
            "S E"
        ]
        maze = self._create_maze(grid)
        plan = plan_optimal_resource_path(maze)
        self.assertEqual(plan.walk_path, [(0, 0), (0, 1), (0, 2)])
        self.assertEqual(plan.max_resource, 0)

    def test_collect_gold(self):
        grid = [
            "S G",
            "# #",
            "E  "
        ]
        maze = self._create_maze(grid)
        plan = plan_optimal_resource_path(maze)
        
        # 必须走 S(0,0) -> (0,1) -> G(0,2) -> (1,2) -> (2,2) -> (2,1) -> E(2,0)
        self.assertIn((0, 2), plan.walk_path, "Path should include the gold cell")
        self.assertGreater(plan.max_resource, 0, "Resource gain should be positive")
        self.assertEqual(len(plan.resource_cells_in_order), 1)

    def test_avoid_trap_if_possible(self):
        grid = [
            "STE",
            "   " 
        ]
        maze = self._create_maze(grid)
        plan = plan_optimal_resource_path(maze)
        
        # Path 1: S -> T -> E (有扣分)
        # Path 2: S -> (1,0) -> (1,1) -> (1,2) -> E (无扣分)
        # 算法应优先保证资源价值最大化（0 > TRAP_VALUE），因此会选择绕路
        self.assertNotIn((0, 1), plan.walk_path, "Path should avoid the trap")
        self.assertEqual(plan.max_resource, 0)

    def test_must_step_trap(self):
        grid = [
            "STE"
        ]
        maze = self._create_maze(grid)
        plan = plan_optimal_resource_path(maze)
        
        # 只有一条路，必须踩陷阱
        self.assertIn((0, 1), plan.walk_path)
        self.assertLess(plan.max_resource, 0, "Resource gain should be negative")

    def test_shortest_path_preferred(self):
        # 当两条路资源收益相同时，应该选择较短的那条
        grid = [
            "S   E",
            "     "
        ]
        maze = self._create_maze(grid)
        plan = plan_optimal_resource_path(maze)
        
        # 收益均为 0，应选择直线走法 (长度5)
        expected_path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        self.assertEqual(plan.walk_path, expected_path)

if __name__ == '__main__':
    unittest.main()
