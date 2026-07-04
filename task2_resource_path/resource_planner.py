from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

from common.models import Cell, MazeGame, Position, StepFrame
from common.pathing import shortest_path
from common.rules import GOLD_VALUE, TRAP_VALUE
try:
    from resource_registry import MazeResourceRegistry
except ModuleNotFoundError:
    from task2_resource_path.resource_registry import MazeResourceRegistry


@dataclass
class ResourcePlan:
    max_resource: int
    walk_path: List[Position]
    resource_cells_in_order: List[Position]
    main_path: List[Position]
    branch_gains: Dict[str, int]
    frames: List[StepFrame]


def _cell_first_gain(maze: MazeGame, pos: Position) -> int:
    cell = maze.grid[pos[0]][pos[1]]
    if cell == Cell.GOLD.value:
        return GOLD_VALUE
    if cell == Cell.TRAP.value:
        return TRAP_VALUE
    return 0


def _cell_repeat_gain(maze: MazeGame, pos: Position) -> int:
    return 0


def _children_off_path(maze: MazeGame, pos: Position, path_prev: Position | None, path_next: Position | None) -> List[Position]:
    out: List[Position] = []
    for nxt in maze.neighbors(pos):
        if nxt != path_prev and nxt != path_next:
            out.append(nxt)
    return out


def _branch_key(parent: Position, child: Position) -> str:
    return f"{parent}->{child}"


def plan_optimal_resource_path(maze: MazeGame) -> ResourcePlan:
    start = maze.find_unique(Cell.START.value)
    exit_pos = maze.find_unique(Cell.EXIT.value)
    main_path = shortest_path(maze, start, exit_pos)

    registry = MazeResourceRegistry(maze)
    K = registry.get_total_resources_count()

    if K == 0:
        return ResourcePlan(0, main_path, [], main_path, {}, [])

    def evaluate_efficiency_score(gain: int, steps: float) -> float:
        if steps == 0:
            return float(gain)
        return gain / steps if gain >= 0 else gain * steps

    def evaluate_accumulated_gain_from_status(collected_status: int) -> int:
        total = 0
        current = collected_status
        while current:
            bit = current & -current
            total += registry.get_resource_value(bit.bit_length() - 1)
            current -= bit
        return total

    # 第一阶段：图论抽象 (Graph Abstraction)
    START_IDX = K
    EXIT_IDX = K + 1
    poi_positions = registry.all_resource_positions + [start, exit_pos]
    NUM_POI = K + 2

    def build_direct_paths_between_pois() -> Tuple[List[List[float]], List[List[List[Position]]]]:
        dist_matrix = [[float('inf')] * NUM_POI for _ in range(NUM_POI)]
        paths_matrix = [[[] for _ in range(NUM_POI)] for _ in range(NUM_POI)]

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

        def reconstruct_path_to(pos: Position, search_tree: Dict[Position, Position | None]) -> List[Position]:
            path: List[Position] = []
            cursor: Tuple[int, int] | None = pos
            while cursor is not None:
                path.append(cursor)
                cursor = search_tree[cursor]
            path.reverse()
            return path

        for i, start_p in enumerate(poi_positions):
            register_direct_path_between(i, i, [start_p])
            direct_search_tree = find_direct_paths_from(start_p)
            
            for j, end_p in enumerate(poi_positions):
                if i != j and end_p in direct_search_tree:
                    path = reconstruct_path_to(end_p, direct_search_tree)
                    register_direct_path_between(i, j, path)
                    
        return dist_matrix, paths_matrix

    dist, path_between = build_direct_paths_between_pois()

    def get_direct_distance_between(u: int, v: int) -> float:
        return dist[u][v]

    def has_direct_connection(u: int, v: int) -> bool:
        return dist[u][v] != float('inf')
        
    def initial_status_of_start() -> int:
        status = 0
        idx = registry.get_resource_id_at(start)
        if idx is not None:
            status |= (1 << idx)
        return status

    def is_resource_collected_in_status(status: int, resource_idx: int) -> bool:
        return bool(status & (1 << resource_idx))

    def collect_resource_in_status(status: int, resource_idx: int) -> int:
        return status | (1 << resource_idx)

    base_dist = [[float('inf')] * NUM_POI for _ in range(NUM_POI)]
    base_next_hop = [[-1] * NUM_POI for _ in range(NUM_POI)]
    for i in range(NUM_POI):
        base_dist[i][i] = 0
        base_next_hop[i][i] = i
        for j in range(NUM_POI):
            if i != j and has_direct_connection(i, j):
                base_dist[i][j] = get_direct_distance_between(i, j)
                base_next_hop[i][j] = j

    floyd_cache: Dict[int, Tuple[List[List[float]], List[List[int]]]] = {}

    def get_floyd_for_status(status: int) -> Tuple[List[List[float]], List[List[int]]]:
        if status in floyd_cache:
            return floyd_cache[status]
            
        D = [row[:] for row in base_dist]
        N = [row[:] for row in base_next_hop]
        
        allowed_k = []
        for k in range(NUM_POI):
            if k < K and is_resource_collected_in_status(status, k):
                allowed_k.append(k)
            elif k >= K:
                allowed_k.append(k)
                
        for k in allowed_k:
            for i in range(NUM_POI):
                if D[i][k] == float('inf'):
                    continue
                for j in range(NUM_POI):
                    if D[k][j] == float('inf'):
                        continue
                    new_dist = D[i][k] + D[k][j]
                    if new_dist < D[i][j]:
                        D[i][j] = new_dist
                        N[i][j] = N[i][k]
                        
        floyd_cache[status] = (D, N)
        return D, N

    # 第二阶段：TSP 状压 DP 结合分支限界与 Beam Search
    gold_indices = [i for i in range(K) if registry.get_resource_value(i) > 0]
    gold_values = [registry.get_resource_value(i) for i in gold_indices]

    def get_optimistic_remaining_gain(status: int) -> int:
        remaining_gain = 0
        for i, val in zip(gold_indices, gold_values):
            if not (status & (1 << i)):
                remaining_gain += val
        return remaining_gain
        
    def get_manhattan_distance(p1: Position, p2: Position) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

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
                # 使用启发式估算效率排序：(当前收益+乐观剩余收益) / (当前步数+到终点的曼哈顿距离)
                sorted_candidates = sorted(
                    self._states.keys(),
                    key=lambda s: (
                        evaluate_efficiency_score(
                            evaluate_accumulated_gain_from_status(s[0]) + get_optimistic_remaining_gain(s[0]),
                            self._states[s] + get_manhattan_distance(poi_positions[s[1]], exit_pos)
                        ),
                        -self._states[s]
                    ),
                    reverse=True
                )
                self._states = {s: self._states[s] for s in sorted_candidates[:beam_width]}

        def is_empty(self) -> bool:
            return len(self._states) == 0

    initial_status = initial_status_of_start()
    current_level_pool = LevelStatePool()
    
    start_state = (initial_status, START_IDX)
    current_level_pool.add_or_relax_state(start_state, 0.0)
    record_shortest_steps(start_state, 0.0)

    best_exit_efficiency = -float('inf')
    best_exit_state_info = None

    def update_global_best_if_can_exit(status: int, endpoint: int, cost: float) -> None:
        nonlocal best_exit_efficiency, best_exit_state_info
        D, N = get_floyd_for_status(status)
        dist_to_exit = D[endpoint][EXIT_IDX]
        if dist_to_exit != float('inf'):
            exit_cost = cost + dist_to_exit
            
            final_status = status
            exit_res_idx = registry.get_resource_id_at(exit_pos)
            if exit_res_idx is not None:
                final_status = collect_resource_in_status(final_status, exit_res_idx)
                
            gain = evaluate_accumulated_gain_from_status(final_status)
            eff = evaluate_efficiency_score(gain, exit_cost)
            
            if eff > best_exit_efficiency:
                best_exit_efficiency = eff
                best_exit_state_info = (status, endpoint, exit_cost)

    update_global_best_if_can_exit(initial_status, START_IDX, 0.0)

    BEAM_WIDTH = 128

    def should_prune_by_branch_and_bound(status: int, endpoint: int, cost: float) -> bool:
        opt_gain = evaluate_accumulated_gain_from_status(status) + get_optimistic_remaining_gain(status)
        min_steps = cost + get_manhattan_distance(poi_positions[endpoint], exit_pos)
        opt_eff = evaluate_efficiency_score(opt_gain, min_steps)
        return opt_eff <= best_exit_efficiency

    # 最多拓展 K 层
    for level in range(K + 1):
        next_level_pool = LevelStatePool()

        for (status, endpoint), cost in current_level_pool.get_all_states():
            if should_prune_by_branch_and_bound(status, endpoint, cost):
                continue

            D, N = get_floyd_for_status(status)

            for next_resource in range(K):
                if not is_resource_collected_in_status(status, next_resource):
                    dist_to_next = D[endpoint][next_resource]
                    if dist_to_next != float('inf'):
                        next_status = collect_resource_in_status(status, next_resource)
                        next_cost = cost + dist_to_next
                        next_state = (next_status, next_resource)

                        if next_level_pool.add_or_relax_state(next_state, next_cost):
                            record_shortest_steps(next_state, next_cost)
                            record_predecessor(next_state, (status, endpoint))
                            update_global_best_if_can_exit(next_status, next_resource, next_cost)

        next_level_pool.keep_top_k_candidates(BEAM_WIDTH)
        current_level_pool = next_level_pool

        if current_level_pool.is_empty():
            break

    if best_exit_state_info is None:
        raise ValueError(f"No path from {start} to {exit_pos}")

    best_status, best_endpoint, total_cost = best_exit_state_info

    # 宏观状态回溯
    macro_sequence = []
    curr_state: Tuple[int, int] | None = (best_status, best_endpoint)
    while curr_state is not None:
        macro_sequence.append(curr_state)
        curr_state = parent_state.get(curr_state)
    macro_sequence.reverse()

    full_poi_sequence = [START_IDX]
    for i in range(len(macro_sequence) - 1):
        status_i, u = macro_sequence[i]
        _, v = macro_sequence[i+1]
        
        D, N = get_floyd_for_status(status_i)
        
        curr_poi = u
        while curr_poi != v:
            curr_poi = N[curr_poi][v]
            full_poi_sequence.append(curr_poi)
            
    # 最后走到 EXIT
    if full_poi_sequence[-1] != EXIT_IDX:
        D, N = get_floyd_for_status(best_status)
        curr_poi = best_endpoint
        while curr_poi != EXIT_IDX:
            curr_poi = N[curr_poi][EXIT_IDX]
            full_poi_sequence.append(curr_poi)

    def reconstruct_walk_path_from(sequence: List[int]) -> List[Position]:
        if len(sequence) == 1:
            return [poi_positions[sequence[0]]]
        path: List[Position] = []
        for idx in range(len(sequence) - 1):
            u = sequence[idx]
            v = sequence[idx + 1]
            segment = path_between[u][v]
            if not path:
                path.extend(segment)
            else:
                path.extend(segment[1:])
        return path

    walk_path = reconstruct_walk_path_from(full_poi_sequence)

    resource_cells_in_order = []
    for pos in walk_path:
        if registry.has_resource_at(pos) and (not resource_cells_in_order or resource_cells_in_order[-1] != pos):
            if pos not in resource_cells_in_order:
                resource_cells_in_order.append(pos)

    # 包含 Exit 的计分
    final_status = best_status
    exit_res_idx = registry.get_resource_id_at(exit_pos)
    if exit_res_idx is not None:
        final_status = collect_resource_in_status(final_status, exit_res_idx)
    max_gain = evaluate_accumulated_gain_from_status(final_status)

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
            gain_here = _cell_first_gain(maze, pos)
            running_resource += gain_here
        append_frame(
            walk_path[: index + 1],
            pos,
            f"移动到 {pos}，本格首次触发收益 {gain_here}，累计资源 {running_resource}",
        )

    return ResourcePlan(
        max_resource=max_gain,
        walk_path=walk_path,
        resource_cells_in_order=resource_cells_in_order,
        main_path=main_path,
        branch_gains=branch_gains,
        frames=frames,
    )


def plan_global_optimal_collection(maze: MazeGame) -> ResourcePlan:
    """
    不考虑特定起点和终点，寻找全局最优的资源收集路径（类似 TSP / 吃豆人模式）。
    利用 图论抽象 + 状态压缩 DP 极大提升搜索效率。
    """
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

    # 第一阶段：提取资源图 (Graph Abstraction)
    # 计算所有资源点之间的最短直达距离与物理路径，建立精简的邻接距离图
    def build_direct_paths_between_resources() -> Tuple[List[List[float]], List[List[List[Position]]]]:
        dist_matrix = [[float('inf')] * K for _ in range(K)]
        paths_matrix = [[[] for _ in range(K)] for _ in range(K)]

        def register_direct_path_between(u: int, v: int, path: List[Position]) -> None:
            # 在邻接距离图中，注册记录两个资源点之间的直通物理轨迹以及对应的步数开销
            dist_matrix[u][v] = len(path) - 1
            paths_matrix[u][v] = path

        # 判定是否在探路时遇到了除了起点之外的其它资源点
        # 技术实现：如果是另一个资源点，则根据算法规则需要停止向外延伸，以保证路径的直达性
        def is_another_resource_besides(pos: Position, start_pos: Position) -> bool:
            return pos != start_pos and registry.has_resource_at(pos)

        # 通过 BFS 寻找从某个资源点出发、不穿越其它资源点的全部“直达”路径树结构
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

        # 判定在直达探路树中是否能到达目标位置
        def is_reachable(pos: Position, search_tree: Dict[Position, Position | None]) -> bool:
            return pos in search_tree

        # 从直达探路树中回溯重建出从起点到目标位置的直通物理路径
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
        # 获取两个资源点之间的直连步数距离
        return dist[u][v]

    def get_direct_path_between(u: int, v: int) -> List[Position]:
        # 获取两个资源点之间的直连物理轨迹段
        return path_between[u][v]

    def has_direct_connection(u: int, v: int) -> bool:
        # 判定两个资源点之间是否存在不穿越其它资源点的直连边
        return dist[u][v] != float('inf')

    def initial_status_of(resource_idx: int) -> int:
        # 获取仅收集了单个指定资源点时的初始状态数值表示
        return 1 << resource_idx

    def is_resource_collected_in_status(status: int, resource_idx: int) -> bool:
        # 判定在当前收集状态中，指定资源点是否已被收集
        return bool(status & (1 << resource_idx))

    def collect_resource_in_status(status: int, resource_idx: int) -> int:
        return status | (1 << resource_idx)

    # 引入 Floyd-Warshall 缓存与预计算逻辑
    base_dist = [[float('inf')] * K for _ in range(K)]
    base_next_hop = [[-1] * K for _ in range(K)]
    for i in range(K):
        base_dist[i][i] = 0
        base_next_hop[i][i] = i
        for j in range(K):
            if i != j and has_direct_connection(i, j):
                base_dist[i][j] = get_direct_distance_between(i, j)
                base_next_hop[i][j] = j

    floyd_cache: Dict[int, Tuple[List[List[float]], List[List[int]]]] = {}

    def get_floyd_for_status(status: int) -> Tuple[List[List[float]], List[List[int]]]:
        if status in floyd_cache:
            return floyd_cache[status]
            
        D = [row[:] for row in base_dist]
        N = [row[:] for row in base_next_hop]
        
        allowed_k = [k for k in range(K) if is_resource_collected_in_status(status, k)]
                
        for k in allowed_k:
            for i in range(K):
                if D[i][k] == float('inf'):
                    continue
                for j in range(K):
                    if D[k][j] == float('inf'):
                        continue
                    new_dist = D[i][k] + D[k][j]
                    if new_dist < D[i][j]:
                        D[i][j] = new_dist
                        N[i][j] = N[i][k]
                        
        floyd_cache[status] = (D, N)
        return D, N

    def get_status_space_upper_bound() -> int:
        return 1 << K

    # 第二阶段：TSP 状态压缩 DP 结合分支限界与 Beam Search
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

            D, N = get_floyd_for_status(status)

            for next_resource in range(K):
                if not is_resource_collected_in_status(status, next_resource):
                    dist_to_next = D[endpoint][next_resource]
                    if dist_to_next != float('inf'):
                        next_status = collect_resource_in_status(status, next_resource)
                        next_cost = cost + dist_to_next
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

    macro_sequence = []
    curr: Tuple[int, int] | None = best_state
    while curr is not None:
        macro_sequence.append(curr)
        curr = parent_state.get(curr)
    macro_sequence.reverse()

    full_poi_sequence = []
    if macro_sequence:
        full_poi_sequence.append(macro_sequence[0][1])
        for i in range(len(macro_sequence) - 1):
            status_i, u = macro_sequence[i]
            _, v = macro_sequence[i+1]
            
            D, N = get_floyd_for_status(status_i)
            
            curr_poi = u
            while curr_poi != v:
                curr_poi = N[curr_poi][v]
                full_poi_sequence.append(curr_poi)

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

    walk_path = reconstruct_walk_path_from(full_poi_sequence)

    resource_cells_in_order = []
    for pos in walk_path:
        if registry.has_resource_at(pos) and (not resource_cells_in_order or resource_cells_in_order[-1] != pos):
            if pos not in resource_cells_in_order:
                resource_cells_in_order.append(pos)

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


