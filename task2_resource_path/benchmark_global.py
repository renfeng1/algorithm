import sys
import os
import json
import time
from typing import Dict, List, Sequence, Set, Tuple
from collections import deque
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import MazeGame, Cell, Position, StepFrame
from common.pathing import shortest_path
from common.rules import GOLD_VALUE, TRAP_VALUE
from resource_registry import MazeResourceRegistry
from resource_planner import plan_global_optimal_collection, ResourcePlan

def _cell_first_gain(maze: MazeGame, pos: Position) -> int:
    cell = maze.grid[pos[0]][pos[1]]
    if cell == Cell.GOLD.value:
        return GOLD_VALUE
    if cell == Cell.TRAP.value:
        return TRAP_VALUE
    return 0

def plan_global_optimal_collection_old(maze: MazeGame) -> ResourcePlan:
    registry = MazeResourceRegistry(maze)
    K = registry.get_total_resources_count()
    
    if K == 0:
        return ResourcePlan(0, [], [], [], {}, [])

    def evaluate_accumulated_gain_from_status(collected_status: int) -> int:
        total = 0
        current = collected_status
        while current:
            bit = current & -current
            total += registry.get_resource_value(bit.bit_length() - 1)
            current -= bit
        return total

    def build_direct_paths_between_resources() -> Tuple[List[List[float]], List[List[List[Position]]]]:
        dist_matrix = [[float('inf')] * K for _ in range(K)]
        paths_matrix = [[[] for _ in range(K)] for _ in range(K)]

        def register_direct_path_between(u: int, v: int, path: List[Position]) -> None:
            dist_matrix[u][v] = len(path) - 1
            paths_matrix[u][v] = path

        def is_another_resource_besides(pos: Position, start_pos: Position) -> bool:
            return pos != start_pos and registry.has_resource_at(pos)

        def find_direct_paths_from(start_pos: Position) -> Dict[Position, Position | None]:
            queue = deque([start_pos])
            direct_search_tree = {start_pos: None}
            
            while queue:
                curr = queue.popleft()
                if is_another_resource_besides(curr, start_pos):
                    continue
                    
                for nxt in maze.neighbors(curr):
                    if nxt not in direct_search_tree:
                        direct_search_tree[nxt] = curr
                        queue.append(nxt)
            return direct_search_tree

        def is_reachable(pos: Position, search_tree: Dict[Position, Position | None]) -> bool:
            return pos in search_tree

        def reconstruct_path_to(pos: Position, search_tree: Dict[Position, Position | None]) -> List[Position]:
            path: List[Position] = []
            cursor: Tuple[int, int] | None = pos
            while cursor is not None:
                path.append(cursor)
                cursor = search_tree[cursor]
            path.reverse()
            return path

        for i, start_pos in enumerate(registry.all_resource_positions):
            register_direct_path_between(i, i, [start_pos])
            
            direct_search_tree = find_direct_paths_from(start_pos)
                        
            for j, end_pos in enumerate(registry.all_resource_positions):
                if i != j and is_reachable(end_pos, direct_search_tree):
                    path = reconstruct_path_to(end_pos, direct_search_tree)
                    register_direct_path_between(i, j, path)
        return dist_matrix, paths_matrix

    dist, path_between = build_direct_paths_between_resources()

    def get_direct_distance_between(u: int, v: int) -> float:
        return dist[u][v]

    def get_direct_path_between(u: int, v: int) -> List[Position]:
        return path_between[u][v]

    def has_direct_connection(u: int, v: int) -> bool:
        return dist[u][v] != float('inf')

    def initial_status_of(resource_idx: int) -> int:
        return 1 << resource_idx

    def is_resource_collected_in_status(status: int, resource_idx: int) -> bool:
        return bool(status & (1 << resource_idx))

    def collect_resource_in_status(status: int, resource_idx: int) -> int:
        return status | (1 << resource_idx)

    def get_status_space_upper_bound() -> int:
        return 1 << K

    def filter_positive_reward_resources() -> Tuple[List[int], List[int]]:
        indices = [
            i for i in range(K)
            if registry.get_resource_value(i) > 0
        ]
        values = [registry.get_resource_value(i) for i in indices]
        return indices, values

    gold_indices, gold_values = filter_positive_reward_resources()

    def get_optimistic_remaining_gain(status: int) -> int:
        remaining_gain = 0
        for i, val in zip(gold_indices, gold_values):
            if not (status & (1 << i)):
                remaining_gain += val
        return remaining_gain

    dp: Dict[Tuple[int, int], float] = {}
    parent_state: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def record_shortest_steps(state: Tuple[int, int], steps: float) -> None:
        dp[state] = steps

    def record_predecessor(state: Tuple[int, int], predecessor: Tuple[int, int]) -> None:
        parent_state[state] = predecessor

    class LevelStatePool:
        def __init__(self):
            self._states: Dict[Tuple[int, int], float] = {}
        
        def add_or_relax_state(self, state: Tuple[int, int], cost: float) -> bool:
            if cost < self._states.get(state, float('inf')):
                self._states[state] = cost
                return True
            return False
            
        def get_all_states(self):
            return self._states.items()
            
        def keep_top_k_candidates(self, beam_width: int) -> None:
            if len(self._states) > beam_width:
                sorted_candidates = sorted(
                    self._states.keys(),
                    key=lambda s: (
                        evaluate_accumulated_gain_from_status(s[0]),
                        -self._states[s]
                    ),
                    reverse=True
                )
                self._states = {s: self._states[s] for s in sorted_candidates[:beam_width]}

        def is_empty(self) -> bool:
            return len(self._states) == 0

    current_level_pool = LevelStatePool()
    for i in range(K):
        state = (initial_status_of(i), i)
        current_level_pool.add_or_relax_state(state, 0.0)
        record_shortest_steps(state, 0.0)

    historical_max_gain = -float('inf')
    historical_min_cost = float('inf')
    historical_best_state = None

    def update_global_best(status: int, endpoint: int, cost: float) -> None:
        nonlocal historical_max_gain, historical_min_cost, historical_best_state
        gain = evaluate_accumulated_gain_from_status(status)
        if (gain > historical_max_gain) or (gain == historical_max_gain and cost < historical_min_cost):
            historical_max_gain = gain
            historical_min_cost = cost
            historical_best_state = (status, endpoint)
            
    for state, cost in current_level_pool.get_all_states():
        update_global_best(state[0], state[1], cost)

    BEAM_WIDTH = 256

    def should_prune_by_branch_and_bound(status: int, cost: float) -> bool:
        opt_gain = evaluate_accumulated_gain_from_status(status) + get_optimistic_remaining_gain(status)
        if opt_gain < historical_max_gain:
            return True
        if opt_gain == historical_max_gain and cost >= historical_min_cost:
            return True
        return False

    for level in range(1, K):
        next_level_pool = LevelStatePool()

        for (status, endpoint), cost in current_level_pool.get_all_states():
            if should_prune_by_branch_and_bound(status, cost):
                continue

            for next_resource in range(K):
                if not is_resource_collected_in_status(status, next_resource):
                    if has_direct_connection(endpoint, next_resource):
                        next_status = collect_resource_in_status(status, next_resource)
                        next_cost = cost + get_direct_distance_between(endpoint, next_resource)
                        next_state = (next_status, next_resource)

                        if next_level_pool.add_or_relax_state(next_state, next_cost):
                            record_shortest_steps(next_state, next_cost)
                            record_predecessor(next_state, (status, endpoint))
                            update_global_best(next_status, next_resource, next_cost)

        next_level_pool.keep_top_k_candidates(BEAM_WIDTH)
        current_level_pool = next_level_pool

        if current_level_pool.is_empty():
            break

    best_state = historical_best_state
    if best_state is None:
        return ResourcePlan(0, [], [], [], {}, [])

    max_gain = evaluate_accumulated_gain_from_status(best_state[0])

    def reconstruct_resource_sequence_from(start_state: Tuple[int, int]) -> List[int]:
        sequence: List[int] = []
        curr: Tuple[int, int] | None = start_state
        while curr is not None:
            _, endpoint = curr
            sequence.append(endpoint)
            curr = parent_state.get(curr)
        sequence.reverse()
        return sequence

    resource_sequence = reconstruct_resource_sequence_from(best_state)

    def reconstruct_walk_path_from(sequence: List[int]) -> List[Position]:
        if not sequence:
            return []
        if len(sequence) == 1:
            return [registry.get_resource_position(sequence[0])]
        
        path: List[Position] = []
        for idx in range(len(sequence) - 1):
            u = sequence[idx]
            v = sequence[idx + 1]
            segment = get_direct_path_between(u, v)
            if not path:
                path.extend(segment)
            else:
                path.extend(segment[1:])
        return path

    walk_path = reconstruct_walk_path_from(resource_sequence)
    resource_cells_in_order = [registry.get_resource_position(u) for u in resource_sequence]

    branch_gains = {
        "resource_cells": K,
        "dp_states": len(dp),
        "objective": max_gain
    }

    frames: List[StepFrame] = []
    running_resource = 0
    triggered: Set[Position] = set()

    def append_frame(current_path: List[Position], current: Position, description: str) -> None:
        frames.append(
            StepFrame(
                grid=maze.clone_grid(),
                title=f"Global Resource step {len(frames) + 1}",
                description=description,
                path=current_path[:],
                highlights=[current],
                meta={"resource": running_resource},
            )
        )

    if walk_path:
        start_pos = walk_path[0]
        append_frame([start_pos], start_pos, "空降并开始全局最优寻路")
        
        for index, pos in enumerate(walk_path):
            gain_here = 0
            if registry.has_resource_at(pos) and pos not in triggered:
                triggered.add(pos)
                gain_here = _cell_first_gain(maze, pos)
                running_resource += gain_here
                
            if index > 0 or gain_here != 0:
                append_frame(
                    walk_path[: index + 1],
                    pos,
                    f"移动到 {pos}，本格收益 {gain_here}，累计资源 {running_resource}",
                )

    return ResourcePlan(
        max_resource=max_gain,
        walk_path=walk_path,
        resource_cells_in_order=resource_cells_in_order,
        main_path=[],
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
    
    with open("GLOBAL_COLLECTION_BENCHMARK.md", "w", encoding="utf-8") as out:
        out.write("# 算法重构 全局寻路 (Global Collection) 性能对比及路径记录\n\n")
        
        for f in files:
            path = os.path.join(maze_dir, f)
            if not os.path.exists(path):
                continue
                
            out.write(f"## 测试地图: `{f}`\n")
            maze = create_maze_from_json(path)
            
            # --- OLD ---
            t0 = time.time()
            plan_old = plan_global_optimal_collection_old(maze)
            t_old = time.time() - t0
            
            # --- NEW ---
            t1 = time.time()
            plan_new = plan_global_optimal_collection(maze)
            t_new = time.time() - t1
            
            out.write("### 改动前 (Old Global DP)\n")
            out.write(f"- **最大资源收益**: {plan_old.max_resource}\n")
            out.write(f"- **探索状态数**: {plan_old.branch_gains.get('dp_states', 0)}\n")
            out.write(f"- **执行耗时**: {t_old:.4f} 秒\n")
            out.write(f"- **总步数**: {max(0, len(plan_old.walk_path) - 1)}\n")
            out.write(f"- **完整路径 (序列)**: \n```python\n{plan_old.walk_path}\n```\n\n")
            
            out.write("### 改动后 (New Global DP + Floyd)\n")
            out.write(f"- **最大资源收益**: {plan_new.max_resource}\n")
            out.write(f"- **探索状态数**: {plan_new.branch_gains.get('dp_states', 0)}\n")
            out.write(f"- **执行耗时**: {t_new:.4f} 秒\n")
            out.write(f"- **总步数**: {max(0, len(plan_new.walk_path) - 1)}\n")
            out.write(f"- **完整路径 (序列)**: \n```python\n{plan_new.walk_path}\n```\n\n")
            out.write("---\n")

if __name__ == "__main__":
    run_comparison()
