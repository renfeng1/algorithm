import sys
import os
import json
import time
from typing import Dict, List, Sequence, Set, Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import MazeGame, Cell, Position, StepFrame
from common.pathing import shortest_path
from common.rules import GOLD_VALUE, TRAP_VALUE
from resource_registry import MazeResourceRegistry
from resource_planner import plan_optimal_resource_path, ResourcePlan

def _cell_first_gain(maze: MazeGame, pos: Position) -> int:
    cell = maze.grid[pos[0]][pos[1]]
    if cell == Cell.GOLD.value:
        return GOLD_VALUE
    if cell == Cell.TRAP.value:
        return TRAP_VALUE
    return 0

def plan_optimal_resource_path_old(maze: MazeGame) -> ResourcePlan:
    import heapq

    start = maze.find_unique(Cell.START.value)
    exit_pos = maze.find_unique(Cell.EXIT.value)
    main_path = shortest_path(maze, start, exit_pos)

    registry = MazeResourceRegistry(maze)

    def evaluate_accumulated_gain_from_bitmap(collected_bitmap: int) -> int:
        total = 0
        current = collected_bitmap
        while current:
            bit = current & -current
            total += registry.get_resource_value(bit.bit_length() - 1)
            current -= bit
        return total

    def collected_bitmap_of(state: Tuple[int, int, int]) -> int:
        return state[2]

    def try_trigger_resource_at(current_bitmap: int, pos: Position) -> int:
        idx = registry.get_resource_id_at(pos)
        if idx is not None:
            return current_bitmap | (1 << idx)
        return current_bitmap

    initial_collected_bitmap = try_trigger_resource_at(0, start)

    gold_indices = [
        i for i in range(registry.get_total_resources_count())
        if registry.get_resource_value(i) > 0
    ]
    gold_values = [registry.get_resource_value(i) for i in gold_indices]

    def get_optimistic_remaining_gain(collected_bitmap: int) -> int:
        remaining_gain = 0
        for i, val in zip(gold_indices, gold_values):
            if not (collected_bitmap & (1 << i)):
                remaining_gain += val
        return remaining_gain

    def get_manhattan_distance(p1: Position, p2: Position) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def evaluate_efficiency_score(gain: int, steps: int) -> float:
        if steps == 0:
            return float(gain)
        return gain / steps if gain >= 0 else gain * steps

    BEAM_WIDTH = 128

    initial_state = (start[0], start[1], initial_collected_bitmap)
    initial_score = evaluate_accumulated_gain_from_bitmap(initial_collected_bitmap)
    opt_gain = initial_score + get_optimistic_remaining_gain(initial_collected_bitmap)
    min_remaining_steps = get_manhattan_distance(start, exit_pos)
    opt_efficiency = evaluate_efficiency_score(opt_gain, min_remaining_steps)

    heap = [(-opt_efficiency, 0, -initial_score, start[0], start[1], initial_collected_bitmap)]
    
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int] | None] = {initial_state: None}
    
    depth: Dict[Tuple[int, int, int], int] = {initial_state: 0}

    cell_beams: Dict[Position, List[int]] = {start: [initial_score]}

    best_exit_efficiency = -float('inf')
    best_exit_state = None

    while heap:
        neg_opt_eff, d, neg_score, r, c, collected_bitmap = heapq.heappop(heap)
        current_state = (r, c, collected_bitmap)
        current_score = -neg_score

        if (r, c) == exit_pos:
            actual_eff = evaluate_efficiency_score(current_score, d)
            if actual_eff > best_exit_efficiency:
                best_exit_efficiency = actual_eff
                best_exit_state = current_state
            continue

        opt_gain = current_score + get_optimistic_remaining_gain(collected_bitmap)
        min_remaining_steps = get_manhattan_distance((r, c), exit_pos)
        opt_efficiency = evaluate_efficiency_score(opt_gain, d + min_remaining_steps)
        if opt_efficiency <= best_exit_efficiency:
            continue

        for nxt in maze.neighbors((r, c)):
            next_collected_bitmap = try_trigger_resource_at(collected_bitmap, nxt)
            next_state = (nxt[0], nxt[1], next_collected_bitmap)
            next_depth = d + 1
            next_score = evaluate_accumulated_gain_from_bitmap(next_collected_bitmap)

            if next_state in depth and next_depth >= depth[next_state]:
                continue

            beams = cell_beams.setdefault(nxt, [])
            if len(beams) >= BEAM_WIDTH and next_score <= beams[-1]:
                continue
            
            next_opt_gain = next_score + get_optimistic_remaining_gain(next_collected_bitmap)
            next_min_steps = get_manhattan_distance(nxt, exit_pos)
            next_opt_eff = evaluate_efficiency_score(next_opt_gain, next_depth + next_min_steps)
            if next_opt_eff <= best_exit_efficiency:
                continue

            beams.append(next_score)
            beams.sort(reverse=True)
            if len(beams) > BEAM_WIDTH:
                beams.pop()

            parent[next_state] = current_state
            depth[next_state] = next_depth
            heapq.heappush(heap, (-next_opt_eff, next_depth, -next_score, nxt[0], nxt[1], next_collected_bitmap))

    if best_exit_state is None:
        raise ValueError(f"No path from {start} to {exit_pos}")

    def reconstruct_path_from(exit_state: Tuple[int, int, int]) -> List[Position]:
        path: List[Position] = []
        cursor: Tuple[int, int, int] | None = exit_state
        while cursor is not None:
            path.append((cursor[0], cursor[1]))
            cursor = parent[cursor]
        path.reverse()
        return path

    walk_path = reconstruct_path_from(best_exit_state)

    max_resource = evaluate_accumulated_gain_from_bitmap(collected_bitmap_of(best_exit_state))
    
    branch_gains: Dict[str, int] = {
        "resource_cells": registry.get_total_resources_count(),
        "state_count": len(parent),
        "objective": max_resource,
    }
    
    frames: List[StepFrame] = []
    running_resource = 0
    triggered: Set[Position] = set()
    resource_cells_in_order: List[Position] = []

    def append_frame(current_path: List[Position], current: Position, description: str) -> None:
        frames.append(
            StepFrame(
                grid=maze.clone_grid(),
                title=f"Resource step {len(frames) + 1}",
                description=description,
                path=current_path[:],
                highlights=[current],
                meta={"resource": running_resource},
            )
        )

    append_frame([start], start, "从起点开始分析最优资源路径")
    
    for index, pos in enumerate(walk_path[1:], start=1):
        gain_here = 0
        idx = registry.get_resource_id_at(pos)
        if idx is not None and pos not in triggered:
            triggered.add(pos)
            resource_cells_in_order.append(pos)
            gain_here = _cell_first_gain(maze, pos)
            running_resource += gain_here
        append_frame(
            walk_path[: index + 1],
            pos,
            f"移动到 {pos}，本格首次触发收益 {gain_here}，累计资源 {running_resource}",
        )

    return ResourcePlan(
        max_resource=max_resource,
        walk_path=walk_path,
        resource_cells_in_order=resource_cells_in_order,
        main_path=main_path,
        branch_gains=branch_gains,
        frames=frames,
    )

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

def run_comparison():
    maze_dir = r"d:\algorithm\task2_resource_path\测试数据\测试数据\task2_DP\测试数据\mazes"
    files = ["maze_7_7.json", "maze_15_15_1.json", "maze_15_15_2.json"]
    
    with open("DP_REFACTOR_BENCHMARK.md", "w", encoding="utf-8") as out:
        out.write("# 算法重构 DP 模型性能对比及路径记录\n\n")
        
        for f in files:
            path = os.path.join(maze_dir, f)
            if not os.path.exists(path):
                continue
                
            out.write(f"## 测试地图: `{f}`\n")
            maze = create_maze_from_json(path)
            
            # --- OLD ---
            t0 = time.time()
            plan_old = plan_optimal_resource_path_old(maze)
            t_old = time.time() - t0
            
            # --- NEW ---
            t1 = time.time()
            plan_new = plan_optimal_resource_path(maze)
            t_new = time.time() - t1
            
            out.write("### 改动前 (Old A* + Beam Search)\n")
            out.write(f"- **最大资源收益**: {plan_old.max_resource}\n")
            out.write(f"- **探索状态数**: {plan_old.branch_gains.get('state_count', 0)}\n")
            out.write(f"- **执行耗时**: {t_old:.4f} 秒\n")
            out.write(f"- **总步数**: {len(plan_old.walk_path) - 1}\n")
            out.write(f"- **完整路径 (序列)**: \n```python\n{plan_old.walk_path}\n```\n\n")
            
            out.write("### 改动后 (New TSP-DP + Beam Search)\n")
            out.write(f"- **最大资源收益**: {plan_new.max_resource}\n")
            out.write(f"- **探索状态数**: {plan_new.branch_gains.get('dp_states', 0)}\n")
            out.write(f"- **执行耗时**: {t_new:.4f} 秒\n")
            out.write(f"- **总步数**: {len(plan_new.walk_path) - 1}\n")
            out.write(f"- **完整路径 (序列)**: \n```python\n{plan_new.walk_path}\n```\n\n")
            out.write("---\n")

if __name__ == "__main__":
    run_comparison()
