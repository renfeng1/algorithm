from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple, NamedTuple

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


class AtomicRoutingNetwork:
    """
    封装图论抽象阶段的原子路径网络。
    记录并查询各关键点（POI）之间不跨越其他关键点的安全直达路径。
    """
    def __init__(self, num_nodes: int):
        self._minimum_steps = [[float('inf')] * num_nodes for _ in range(num_nodes)]
        self._physical_footprints = [[[] for _ in range(num_nodes)] for _ in range(num_nodes)]

    def establish_safe_passage(self, start_node: int, end_node: int, footprints: List[Position]) -> None:
        """确立两个关键点之间的安全通行路线及开销"""
        self._minimum_steps[start_node][end_node] = len(footprints) - 1
        self._physical_footprints[start_node][end_node] = footprints

    def can_travel_safely_without_interruption(self, start_node: int, end_node: int) -> bool:
        """评估两点之间是否存在不被打断（不跨越其他关键点）的安全路线"""
        return self._minimum_steps[start_node][end_node] != float('inf')

    def assess_travel_cost(self, start_node: int, end_node: int) -> float:
        """评估安全通行的最少步数开销"""
        return self._minimum_steps[start_node][end_node]

    def retrieve_travel_footprints(self, start_node: int, end_node: int) -> List[Position]:
        """提取安全通行所需的完整物理移动轨迹"""
        return self._physical_footprints[start_node][end_node]


class DynamicTopologyManager:
    """
    负责在 AtomicRoutingNetwork 的基础上，根据玩家当前已收集的资源（作为中转站），
    动态通过 Floyd-Warshall 算法推演出全图的最短路径和路由表，并对计算结果进行缓存。
    """
    def __init__(self, routing_network: 'AtomicRoutingNetwork', num_nodes: int, num_resources: int):
        self.routing_network = routing_network
        self.num_nodes = num_nodes
        self.num_resources = num_resources
        self.floyd_cache: Dict[int, Tuple[List[List[float]], List[List[int]]]] = {}
        self.base_distances, self.base_next_hop_table = self._initialize_floyd_base_matrices()

    def _initialize_floyd_base_matrices(self) -> Tuple[List[List[float]], List[List[int]]]:
        initial_distances = [[float('inf')] * self.num_nodes for _ in range(self.num_nodes)]
        initial_next_hop_table = [[-1] * self.num_nodes for _ in range(self.num_nodes)]

        def record_zero_cost_self_loop(node: int) -> None:
            initial_distances[node][node] = 0
            initial_next_hop_table[node][node] = node

        def record_direct_neighbor_hop(start_node: int, end_node: int) -> None:
            initial_distances[start_node][end_node] = self.routing_network.assess_travel_cost(start_node, end_node)
            initial_next_hop_table[start_node][end_node] = end_node

        for i in range(self.num_nodes):
            record_zero_cost_self_loop(i)
            for j in range(self.num_nodes):
                if i != j and self.routing_network.can_travel_safely_without_interruption(i, j):
                    record_direct_neighbor_hop(i, j)
        return initial_distances, initial_next_hop_table

    def build_routing_network_via_collected_transit_nodes(self, status: int) -> Tuple[List[List[float]], List[List[int]]]:
        if status in self.floyd_cache:
            return self.floyd_cache[status]
            
        distances = [row[:] for row in self.base_distances]
        next_hop_table = [row[:] for row in self.base_next_hop_table]
        
        def identify_allowed_transit_nodes() -> List[int]:
            transit_nodes = []
            for node in range(self.num_nodes):
                if node < self.num_resources and bool(status & (1 << node)):
                    transit_nodes.append(node)
                elif node >= self.num_resources:
                    transit_nodes.append(node)
            return transit_nodes
                
        def update_route_if_shorter_via_transit(start_node: int, end_node: int, transit_node: int) -> None:
            dist_via_transit = distances[start_node][transit_node] + distances[transit_node][end_node]
            if dist_via_transit < distances[start_node][end_node]:
                distances[start_node][end_node] = dist_via_transit
                next_hop_table[start_node][end_node] = next_hop_table[start_node][transit_node]

        def relax_paths_via_transit_node(transit_node: int) -> None:
            for start_node in range(self.num_nodes):
                if distances[start_node][transit_node] == float('inf'):
                    continue
                for end_node in range(self.num_nodes):
                    if distances[transit_node][end_node] != float('inf'):
                        update_route_if_shorter_via_transit(start_node, end_node, transit_node)

        allowed_transit_nodes = identify_allowed_transit_nodes()
        for transit_node in allowed_transit_nodes:
            relax_paths_via_transit_node(transit_node)
                        
        self.floyd_cache[status] = (distances, next_hop_table)
        return distances, next_hop_table




class DPState(NamedTuple):
    """
    表示状态压缩 DP 搜索过程中的一个状态节点。
    """
    collected_status: int
    current_node: int

def is_resource_collected_in_status(status: int, resource_idx: int) -> bool:
    return bool(status & (1 << resource_idx))

def collect_resource_in_status(status: int, resource_idx: int) -> int:
    return status | (1 << resource_idx)

class StateTransitionHistory:
    """
    记录状态压缩 DP 搜索过程中的状态转移历史（前驱节点），
    用于在找到最优解后逆向回溯出完整的访问序列（Macro Sequence）。
    """
    def __init__(self):
        self._parents: Dict[DPState, DPState] = {}

    def record_predecessor(self, current_state: DPState, previous_state: DPState) -> None:
        self._parents[current_state] = previous_state

    def reconstruct_macro_sequence(self, end_state: DPState) -> List[DPState]:
        sequence = []
        curr: DPState | None = end_state
        while curr is not None:
            sequence.append(curr)
            curr = self._parents.get(curr)
        sequence.reverse()
        return sequence

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
    START_IDX = K#编号K分配給起点
    EXIT_IDX = K + 1#编号K+1分配给终点
    poi_positions = registry.all_resource_positions + [start, exit_pos]#0-K分配给资源点（金币或者陷阱）
    NUM_POI = K + 2

    def discover_atomic_routing_network() -> AtomicRoutingNetwork:
        network = AtomicRoutingNetwork(NUM_POI)

        def is_interrupted_by_other_resource(pos: Position, origin_pos: Position) -> bool:
            return pos != origin_pos and registry.has_resource_at(pos)

        def build_direct_connectivity_tree_from(origin_pos: Position) -> Dict[Position, Position | None]:
            queue = deque([origin_pos])
            discovery_tree = {origin_pos: None}
            
            def expand_unvisited_neighbors(current_pos: Position) -> None:
                for neighbor in maze.neighbors(current_pos):
                    if neighbor not in discovery_tree:
                        discovery_tree[neighbor] = current_pos
                        queue.append(neighbor)

            while queue:
                curr = queue.popleft()
                if is_interrupted_by_other_resource(curr, origin_pos):
                    continue
                    
                expand_unvisited_neighbors(curr)
            return discovery_tree

        def trace_footprints_to(destination: Position, connectivity_tree: Dict[Position, Position | None]) -> List[Position]:
            footprints: List[Position] = []
            cursor: Tuple[int, int] | None = destination
            while cursor is not None:
                footprints.append(cursor)
                cursor = connectivity_tree[cursor]
            footprints.reverse()
            return footprints

        for origin_idx, origin_pos in enumerate(poi_positions):
            network.establish_safe_passage(origin_idx, origin_idx, [origin_pos])
            connectivity_tree = build_direct_connectivity_tree_from(origin_pos)
            
            for dest_idx, dest_pos in enumerate(poi_positions):
                if origin_idx != dest_idx and dest_pos in connectivity_tree:
                    footprints = trace_footprints_to(dest_pos, connectivity_tree)
                    network.establish_safe_passage(origin_idx, dest_idx, footprints)
                    
        return network

    routing_network = discover_atomic_routing_network()

    def initial_status_of_start() -> int:
        status = 0
        idx = registry.get_resource_id_at(start)
        if idx is not None:
            status |= (1 << idx)
        return status

    topology_manager = DynamicTopologyManager(routing_network, NUM_POI, K)

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

    transition_history = StateTransitionHistory()

    class ResourceCountStatePool:
        def __init__(self):
            # 记录在当前搜索层级下，到达某一特定状态（已收集资源+当前位置）所耗费的最短步数(cost)，即 DP 中的 f(S, i)
            self._min_steps_to_reach_state: Dict[DPState, float] = {}
        
        def add_or_relax_state(self, state: DPState, steps: float) -> bool:
            if steps < self._min_steps_to_reach_state.get(state, float('inf')):
                self._min_steps_to_reach_state[state] = steps
                return True
            return False
            
        def get_all_states(self):
            return self._min_steps_to_reach_state.items()
            
        def keep_top_k_candidates(self, beam_width: int) -> None:
            if len(self._min_steps_to_reach_state) > beam_width:
                # 使用启发式估算效率排序：(当前收益+乐观剩余收益) / (当前步数+到终点的曼哈顿距离)
                sorted_candidates = sorted(
                    self._min_steps_to_reach_state.keys(),
                    key=lambda s: (
                        evaluate_efficiency_score(
                            evaluate_accumulated_gain_from_status(s.collected_status) + get_optimistic_remaining_gain(s.collected_status),
                            self._min_steps_to_reach_state[s] + get_manhattan_distance(poi_positions[s.current_node], exit_pos)
                        ),
                        -self._min_steps_to_reach_state[s]
                    ),
                    reverse=True
                )
                self._min_steps_to_reach_state = {s: self._min_steps_to_reach_state[s] for s in sorted_candidates[:beam_width]}

        def is_empty(self) -> bool:
            return len(self._min_steps_to_reach_state) == 0

    initial_status = initial_status_of_start()
    current_count_pool = ResourceCountStatePool()
    
    start_state = DPState(initial_status, START_IDX)
    current_count_pool.add_or_relax_state(start_state, 0.0)

    best_exit_efficiency = -float('inf')
    best_exit_state_info = None

    def get_safe_distance_to_exit(status: int, endpoint: int) -> float:
        distances, _ = topology_manager.build_routing_network_via_collected_transit_nodes(status)
        return distances[endpoint][EXIT_IDX]

    def include_exit_resource_if_any(current_status: int) -> int:
        exit_res_idx = registry.get_resource_id_at(exit_pos)
        if exit_res_idx is not None:
            return collect_resource_in_status(current_status, exit_res_idx)
        return current_status

    def record_if_best_exit_strategy(status: int, endpoint: int, exit_steps: float, final_status: int) -> None:
        nonlocal best_exit_efficiency, best_exit_state_info
        gain = evaluate_accumulated_gain_from_status(final_status)
        eff = evaluate_efficiency_score(gain, exit_steps)
        if eff > best_exit_efficiency:
            best_exit_efficiency = eff
            best_exit_state_info = (status, endpoint, exit_steps)

    def evaluate_and_record_dash_to_exit_strategy(status: int, endpoint: int, steps: float) -> None:
        dist_to_exit = get_safe_distance_to_exit(status, endpoint)
        if dist_to_exit == float('inf'):
            return
            
        exit_steps = steps + dist_to_exit
        final_status = include_exit_resource_if_any(status)
        record_if_best_exit_strategy(status, endpoint, exit_steps, final_status)

    evaluate_and_record_dash_to_exit_strategy(initial_status, START_IDX, 0.0)

    BEAM_WIDTH = 128

    def should_prune_by_branch_and_bound(status: int, endpoint: int, steps: float) -> bool:
        opt_gain = evaluate_accumulated_gain_from_status(status) + get_optimistic_remaining_gain(status)
        min_steps = steps + get_manhattan_distance(poi_positions[endpoint], exit_pos)
        opt_eff = evaluate_efficiency_score(opt_gain, min_steps)
        return opt_eff <= best_exit_efficiency

    def can_safely_reach_next_resource(distances: List[List[float]], endpoint: int, next_resource: int) -> bool:
        return distances[endpoint][next_resource] != float('inf')

    def expand_next_states_from_current(status: int, endpoint: int, steps: float, distances: List[List[float]], next_count_pool: ResourceCountStatePool) -> None:
        for next_resource in range(K):
            if not is_resource_collected_in_status(status, next_resource):
                if can_safely_reach_next_resource(distances, endpoint, next_resource):
                    next_steps = steps + distances[endpoint][next_resource]
                    next_state = DPState(collect_resource_in_status(status, next_resource), next_resource)

                    if next_count_pool.add_or_relax_state(next_state, next_steps):
                        transition_history.record_predecessor(next_state, DPState(status, endpoint))
                        evaluate_and_record_dash_to_exit_strategy(next_state.collected_status, next_resource, next_steps)

    def execute_layered_dp_search() -> None:
        nonlocal current_count_pool
        for num_collected_resources in range(K + 1):
            next_count_pool = ResourceCountStatePool()

            for (status, endpoint), steps in current_count_pool.get_all_states():
                if should_prune_by_branch_and_bound(status, endpoint, steps):
                    continue

                distances, _ = topology_manager.build_routing_network_via_collected_transit_nodes(status)
                expand_next_states_from_current(status, endpoint, steps, distances, next_count_pool)

            next_count_pool.keep_top_k_candidates(BEAM_WIDTH)
            current_count_pool = next_count_pool

            if current_count_pool.is_empty():
                break

    execute_layered_dp_search()

    if best_exit_state_info is None:
        raise ValueError(f"No path from {start} to {exit_pos}")

    best_status, best_endpoint, total_cost = best_exit_state_info

    # 宏观状态回溯
    macro_sequence = transition_history.reconstruct_macro_sequence(DPState(best_status, best_endpoint))

    full_poi_sequence = [START_IDX]
    for i in range(len(macro_sequence) - 1):
        status_i, u = macro_sequence[i]
        _, v = macro_sequence[i+1]
        
        distances, next_hop_table = topology_manager.build_routing_network_via_collected_transit_nodes(status_i)
        
        curr_poi = u
        while curr_poi != v:
            curr_poi = next_hop_table[curr_poi][v]
            full_poi_sequence.append(curr_poi)
            
    # 最后走到 EXIT
    if full_poi_sequence[-1] != EXIT_IDX:
        distances, next_hop_table = topology_manager.build_routing_network_via_collected_transit_nodes(best_status)
        curr_poi = best_endpoint
        while curr_poi != EXIT_IDX:
            curr_poi = next_hop_table[curr_poi][EXIT_IDX]
            full_poi_sequence.append(curr_poi)

    def reconstruct_walk_path_from(sequence: List[int]) -> List[Position]:
        if len(sequence) == 1:
            return [poi_positions[sequence[0]]]
        path: List[Position] = []
        for idx in range(len(sequence) - 1):
            u = sequence[idx]
            v = sequence[idx + 1]
            segment = routing_network.retrieve_travel_footprints(u, v)
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
        "dp_states": len(transition_history._parents),
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
    def discover_atomic_routing_network() -> AtomicRoutingNetwork:
        network = AtomicRoutingNetwork(K)

        def is_interrupted_by_other_resource(pos: Position, origin_pos: Position) -> bool:
            return pos != origin_pos and registry.has_resource_at(pos)

        def build_direct_connectivity_tree_from(origin_pos: Position) -> Dict[Position, Position | None]:
            queue = deque([origin_pos])
            discovery_tree = {origin_pos: None}
            
            def expand_unvisited_neighbors(current_pos: Position) -> None:
                for neighbor in maze.neighbors(current_pos):
                    if neighbor not in discovery_tree:
                        discovery_tree[neighbor] = current_pos
                        queue.append(neighbor)

            while queue:
                curr = queue.popleft()
                if is_interrupted_by_other_resource(curr, origin_pos):
                    continue
                    
                expand_unvisited_neighbors(curr)
            return discovery_tree

        def trace_footprints_to(destination: Position, connectivity_tree: Dict[Position, Position | None]) -> List[Position]:
            footprints: List[Position] = []
            cursor: Tuple[int, int] | None = destination
            while cursor is not None:
                footprints.append(cursor)
                cursor = connectivity_tree[cursor]
            footprints.reverse()
            return footprints

        for origin_idx, origin_pos in enumerate(registry.all_resource_positions):
            network.establish_safe_passage(origin_idx, origin_idx, [origin_pos])
            connectivity_tree = build_direct_connectivity_tree_from(origin_pos)
            
            for dest_idx, dest_pos in enumerate(registry.all_resource_positions):
                if origin_idx != dest_idx and dest_pos in connectivity_tree:
                    footprints = trace_footprints_to(dest_pos, connectivity_tree)
                    network.establish_safe_passage(origin_idx, dest_idx, footprints)
                    
        return network

    routing_network = discover_atomic_routing_network()

    def initial_status_of(resource_idx: int) -> int:
        return 1 << resource_idx

    topology_manager = DynamicTopologyManager(routing_network, K, K)

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

    transition_history = StateTransitionHistory()

    class ResourceCountStatePool:
        def __init__(self):
            # 记录在当前搜索层级下，到达某一特定状态（已收集资源+当前位置）所耗费的最短步数(cost)，即 DP 中的 f(S, i)
            self._min_steps_to_reach_state: Dict[DPState, float] = {}
        
        def add_or_relax_state(self, state: DPState, steps: float) -> bool:
            if steps < self._min_steps_to_reach_state.get(state, float('inf')):
                self._min_steps_to_reach_state[state] = steps
                return True
            return False
            
        def get_all_states(self):
            return self._min_steps_to_reach_state.items()
            
        def keep_top_k_candidates(self, beam_width: int) -> None:
            if len(self._min_steps_to_reach_state) > beam_width:
                sorted_candidates = sorted(
                    self._min_steps_to_reach_state.keys(),
                    key=lambda s: (
                        evaluate_accumulated_gain_from_status(s.collected_status),
                        -self._min_steps_to_reach_state[s]
                    ),
                    reverse=True
                )
                self._min_steps_to_reach_state = {s: self._min_steps_to_reach_state[s] for s in sorted_candidates[:beam_width]}

        def is_empty(self) -> bool:
            return len(self._min_steps_to_reach_state) == 0

    current_count_pool = ResourceCountStatePool()
    for i in range(K):
        state = DPState(initial_status_of(i), i)
        current_count_pool.add_or_relax_state(state, 0.0)

    historical_max_gain = -float('inf')
    historical_min_steps = float('inf')
    historical_best_state = None

    def update_global_best(status: int, endpoint: int, steps: float) -> None:
        nonlocal historical_max_gain, historical_min_steps, historical_best_state
        gain = evaluate_accumulated_gain_from_status(status)
        if (gain > historical_max_gain) or (gain == historical_max_gain and steps < historical_min_steps):
            historical_max_gain = gain
            historical_min_steps = steps
            historical_best_state = DPState(status, endpoint)

    for state, steps in current_count_pool.get_all_states():
        update_global_best(state.collected_status, state.current_node, steps)

    BEAM_WIDTH = 256

    def should_prune_by_branch_and_bound(status: int, steps: float) -> bool:
        opt_gain = evaluate_accumulated_gain_from_status(status) + get_optimistic_remaining_gain(status)
        if opt_gain < historical_max_gain:
            return True
        if opt_gain == historical_max_gain and cost >= historical_min_cost:
            return True
        return False

    def can_safely_reach_next_resource(distances: List[List[float]], endpoint: int, next_resource: int) -> bool:
        return distances[endpoint][next_resource] != float('inf')

    def expand_next_states_from_current(status: int, endpoint: int, steps: float, distances: List[List[float]], next_count_pool: ResourceCountStatePool) -> None:
        for next_resource in range(K):
            if not is_resource_collected_in_status(status, next_resource):
                if can_safely_reach_next_resource(distances, endpoint, next_resource):
                    next_steps = steps + distances[endpoint][next_resource]
                    next_state = DPState(collect_resource_in_status(status, next_resource), next_resource)

                    if next_count_pool.add_or_relax_state(next_state, next_steps):
                        transition_history.record_predecessor(next_state, DPState(status, endpoint))
                        update_global_best(next_state.collected_status, next_resource, next_steps)

    def execute_layered_dp_search() -> None:
        nonlocal current_count_pool
        for num_collected_resources in range(1, K):
            next_count_pool = ResourceCountStatePool()

            for (status, endpoint), steps in current_count_pool.get_all_states():
                if should_prune_by_branch_and_bound(status, steps):
                    continue

                distances, _ = topology_manager.build_routing_network_via_collected_transit_nodes(status)
                expand_next_states_from_current(status, endpoint, steps, distances, next_count_pool)

            next_count_pool.keep_top_k_candidates(BEAM_WIDTH)
            current_count_pool = next_count_pool

            if current_count_pool.is_empty():
                break

    execute_layered_dp_search()

    best_state = historical_best_state
    if best_state is None:
        return ResourcePlan(0, [], [], [], {}, [])

    max_gain = evaluate_accumulated_gain_from_status(best_state[0])

    macro_sequence = transition_history.reconstruct_macro_sequence(best_state)

    full_poi_sequence = []
    if macro_sequence:
        full_poi_sequence.append(macro_sequence[0][1])
        for i in range(len(macro_sequence) - 1):
            status_i, u = macro_sequence[i]
            _, v = macro_sequence[i+1]
            
            distances, next_hop_table = topology_manager.build_routing_network_via_collected_transit_nodes(status_i)
            
            curr_poi = u
            while curr_poi != v:
                curr_poi = next_hop_table[curr_poi][v]
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
            segment = routing_network.retrieve_travel_footprints(u, v)
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
        "dp_states": len(transition_history._parents),
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


