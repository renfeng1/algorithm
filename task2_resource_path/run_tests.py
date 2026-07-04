import sys
import os
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import MazeGame
from resource_planner import plan_optimal_resource_path

def create_maze_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        grid = data.get('maze', data.get('grid', []))
        return MazeGame(
            grid=grid,
            boss_health=[],
            player_skills=[],
            min_rounds=0,
            coin_consumption=0
        )

def run_tests():
    maze_dir = r"d:\algorithm\task2_resource_path\测试数据\测试数据\task2_DP\测试数据\mazes"
    files = ["maze_7_7.json", "maze_15_15_1.json", "maze_15_15_2.json"]
    
    for f in files:
        path = os.path.join(maze_dir, f)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        print(f"--- Testing {f} ---")
        maze = create_maze_from_json(path)
        start_time = time.time()
        try:
            plan = plan_optimal_resource_path(maze)
            duration = time.time() - start_time
            print(f"Success in {duration:.4f} seconds!")
            print(f"Max Resource Gain: {plan.max_resource}")
            print(f"Walk Path Length: {len(plan.walk_path)}")
            print(f"Resource Cells Hit: {len(plan.resource_cells_in_order)}")
            print(f"Branch Gains Info: {plan.branch_gains}")
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    run_tests()
