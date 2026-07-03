from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

from common.models import Cell, MazeGame, Position, StepFrame
from common.pathing import shortest_path
from common.rules import GOLD_VALUE, TRAP_VALUE
from resource_registry import MazeResourceRegistry


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

    # 业务意图：评估当前资源收集状态的总净收益（吃金币加分，踩陷阱扣分）
    # 技术实现：利用 lowbit (Brian Kernighan 算法) 极速跳过 bitmap 中的 0，仅对为 1 的资源位进行累加
    def evaluate_accumulated_gain_from_bitmap(collected_bitmap: int) -> int:
        total = 0
        current = collected_bitmap
        while current:
            bit = current & -current
            total += registry.get_resource_value(bit.bit_length() - 1)
            current -= bit
        return total

    def collected_bitmap_of(state: Tuple[int, int, int]) -> int:
        """业务意图：从高维搜索状态中，提取出当前已收集资源的位图表示"""
        return state[2]

    # 业务意图：当移动到新坐标时，尝试触发并收集该位置的资源（若存在）
    # 技术实现：若坐标命中资源索引表，则将 bitmap 中对应的二进制位设为 1，并返回更新后的 bitmap
    def try_trigger_resource_at(current_bitmap: int, pos: Position) -> int:
        idx = registry.get_resource_id_at(pos)
        if idx is not None:
            return current_bitmap | (1 << idx)
        return current_bitmap

    initial_collected_bitmap = try_trigger_resource_at(0, start)

    # =========================================================================
    # 状态压缩 DP (State Compression Dynamic Programming) 原理说明：
    #
    # 1. DP 状态定义：
    #    dp[r][c][collected_bitmap] 表示从起点 (start) 出发，到达网格坐标 (r, c)，
    #    且当前已触发的资源状态为 collected_bitmap（按位记录每个资源 ID 是否已被触发）时的【最少移动步数】。
    #
    # 2. DP 状态转移方程：
    #    对于当前位置 (r, c) 的每一个可通行邻居 nxt:
    #    next_bitmap = try_trigger_resource_at(collected_bitmap, nxt)
    #    dp[nxt.r][nxt.c][next_bitmap] = min(dp[nxt.r][nxt.c][next_bitmap], dp[r][c][collected_bitmap] + 1)
    #
    # 3. 算法实现与状态剪枝：
    #    由于图的边权（每次移动）均为 1，我们使用状态空间 (r, c, collected_bitmap) 上的 BFS
    #    来实现该 DP。在无权图 BFS 中，每个状态第一次入队时，其对应的路径深度即为 dp 的最优解（最小步数）。
    #    通过 `next_state in parent` 的去重判断，我们实现了对已计算 DP 状态的剪枝。
    # =========================================================================

    # 1. 状态定义与初始化
    # 每个搜索状态表示为：(行, 列, 当前已收集资源的位图)
    state = (start[0], start[1], initial_collected_bitmap)
    queue = deque([state])
    
    # 状态转移父节点表，用于后续逆向回溯重建最优物理路径
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int] | None] = {state: None}
    
    # 记录每个状态的移动步数（路径长度），用于在收益相同时进行“短路径优先”的仲裁
    depth: Dict[Tuple[int, int, int], int] = {state: 0}
    
    # 收集所有能够成功到达终点的状态候选集
    exit_states: List[Tuple[int, int, int]] = [state] if start == exit_pos else []

    # 2. 状态空间搜索（BFS 遍历）
    while queue:
        r, c, collected_bitmap = queue.popleft()
        current_state = (r, c, collected_bitmap)
        
        # 如果当前状态到达了终点坐标，加入候选集，并停止继续从终点扩展
        if (r, c) == exit_pos:
            exit_states.append(current_state)
            continue

        # 遍历当前位置的相邻网格
        for nxt in maze.neighbors((r, c)):
            # 移动到新网格，尝试收集该位置的资源，获取更新后的资源收集位图
            next_collected_bitmap = try_trigger_resource_at(collected_bitmap, nxt)
            next_state = (nxt[0], nxt[1], next_collected_bitmap)
            
            # 若该状态此前已被遍历过，跳过以避免死循环
            if next_state in parent:
                continue
                
            # 记录新状态的父节点和步数，并入队继续搜索
            parent[next_state] = current_state
            depth[next_state] = depth[current_state] + 1
            queue.append(next_state)

    # 3. 终点选择与最优路径重建
    # 业务意图：评估一条路径的资源收集效率分数（获得的资源 / 走的步数）
    # 特殊处理：对于负收益（踩陷阱），步数越多应该越劣（因此乘以 steps）；对于步数为 0 的情况进行安全规避
    def evaluate_efficiency_score(gain: int, steps: int) -> float:
        if steps == 0:
            return float(gain)
        return gain / steps if gain >= 0 else gain * steps

    # 业务意图：从所有到达终点的可能状态中，查询最优的那个状态（资源收集效率最高，步数最短）
    def query_best_exit_state() -> Tuple[int, int, int] | None:
        if not exit_states:
            return None
        return max(
            exit_states,
            key=lambda s: (
                evaluate_efficiency_score(
                    evaluate_accumulated_gain_from_bitmap(collected_bitmap_of(s)),
                    depth[s]
                ),
                -depth[s]
            )
        )

    best_exit_state = query_best_exit_state()
    if best_exit_state is None:
        raise ValueError(f"No path from {start} to {exit_pos}")

    # 业务意图：根据最优终点状态，沿着状态转移链逆向回溯，重建出顺向的物理网格坐标路径
    def reconstruct_path_from(exit_state: Tuple[int, int, int]) -> List[Position]:
        path: List[Position] = []
        cursor: Tuple[int, int, int] | None = exit_state
        while cursor is not None:
            path.append((cursor[0], cursor[1]))
            cursor = parent[cursor]
        path.reverse()  # 翻转序列使其从起点指向终点
        return path

    walk_path = reconstruct_path_from(best_exit_state)

    # 计算该最优路径下根据收集的位图得到的总资源收益
    max_resource = evaluate_accumulated_gain_from_bitmap(collected_bitmap_of(best_exit_state))
    
    # 统计元信息，供上层日志和性能评估使用
    branch_gains: Dict[str, int] = {
        "resource_cells": registry.get_total_resources_count(),
        "state_count": len(parent),
        "objective": max_resource,
    }
    
    # 4. 生成可视化动作序列帧 (Step Frames)
    frames: List[StepFrame] = []
    running_resource = 0
    triggered: Set[Position] = set()
    resource_cells_in_order: List[Position] = []

    # 辅助函数：向帧序列中追加当前的移动状态快照
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

    # 记录起始帧状态
    append_frame([start], start, "从起点开始分析最优资源路径")
    
    # 顺向遍历计算出每一步的收益与触发记录，并生成可视化帧
    for index, pos in enumerate(walk_path[1:], start=1):
        gain_here = 0
        idx = registry.get_resource_id_at(pos)
        # 如果本格是资源格且在该路径上是首次触发（金币/陷阱只能收集一次）
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


def plan_global_optimal_collection(maze: MazeGame) -> ResourcePlan:
    """
    不考虑特定起点和终点，寻找全局最优的资源收集路径（类似 TSP / 吃豆人模式）。
    利用 图论抽象 + 状态压缩 DP 极大提升搜索效率。
    """
    registry = MazeResourceRegistry(maze)
    K = registry.get_total_resources_count()
    
    if K == 0:
        return ResourcePlan(0, [], [], [], {}, [])

    def evaluate_accumulated_gain_from_bitmap(collected_bitmap: int) -> int:
        total = 0
        current = collected_bitmap
        while current:
            bit = current & -current
            total += registry.get_resource_value(bit.bit_length() - 1)
            current -= bit
        return total

    # 第一阶段：提取资源图 (Graph Abstraction)
    # 业务意图：计算所有资源点之间的最短直达距离与物理路径，建立精简的邻接距离图
    def build_direct_paths_between_resources() -> Tuple[List[List[float]], List[List[List[Position]]]]:
        dist_matrix = [[float('inf')] * K for _ in range(K)]
        paths_matrix = [[[] for _ in range(K)] for _ in range(K)]

        def register_direct_path_between(u: int, v: int, path: List[Position]) -> None:
            # 业务意图：在邻接距离图中，注册记录两个资源点之间的直通物理轨迹以及对应的步数开销
            dist_matrix[u][v] = len(path) - 1
            paths_matrix[u][v] = path

        # 业务意图：判定是否在探路时遇到了除了起点之外的其它资源点
        # 技术实现：如果是另一个资源点，则根据算法规则需要停止向外延伸，以保证路径的直达性
        def is_another_resource_besides(pos: Position, start_pos: Position) -> bool:
            return pos != start_pos and registry.has_resource_at(pos)

        # 业务意图：通过 BFS 寻找从某个资源点出发、不穿越其它资源点的全部“直达”路径树结构
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

        # 业务意图：判定在直达探路树中是否能到达目标位置
        def is_reachable(pos: Position, search_tree: Dict[Position, Position | None]) -> bool:
            return pos in search_tree

        # 业务意图：从直达探路树中回溯重建出从起点到目标位置的直通物理路径
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
        # 业务意图：获取两个资源点之间的直连步数距离
        return dist[u][v]

    def get_direct_path_between(u: int, v: int) -> List[Position]:
        # 业务意图：获取两个资源点之间的直连物理轨迹段
        return path_between[u][v]

    def has_direct_connection(u: int, v: int) -> bool:
        # 业务意图：判定两个资源点之间是否存在不穿越其它资源点的直连边
        return dist[u][v] != float('inf')

    def initial_collected_status_of(resource_idx: int) -> int:
        # 业务意图：获取仅收集了单个指定资源点时的初始状态位图
        return 1 << resource_idx

    def is_resource_collected_in(status_bitmap: int, resource_idx: int) -> bool:
        # 业务意图：判定在当前收集状态位图中，指定资源点是否已被收集
        return bool(status_bitmap & (1 << resource_idx))

    def collect_resource_in(status_bitmap: int, resource_idx: int) -> int:
        # 业务意图：在当前收集状态中加入指定资源点，并返回更新后的收集状态位图
        return status_bitmap | (1 << resource_idx)

    def get_total_status_combinations() -> int:
        # 业务意图：获取所有可能的资源收集状态组合总数 (2 的 K 次方)
        return 1 << K

    # 第二阶段：TSP 状态压缩 DP
    dp: Dict[Tuple[int, int], float] = { 
        (mask, u): float('inf') 
        for mask in range(get_total_status_combinations()) 
        for u in range(K) 
    }
    parent_state: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # 初始状态：可以任意选择一个资源点空降开局，步数为 0
    for i in range(K):
        dp[(initial_collected_status_of(i), i)] = 0

    def try_transition_from_resource(u: int, to_resource: int, under_status: int) -> None:
        # 业务意图：尝试从当前所在的资源点 u 转移到下一个未收集的资源点 to_resource
        # 技术实现：检查连通性并进行 DP 状态的最短路径松弛更新，同时记录回溯父状态
        if not is_resource_collected_in(under_status, to_resource):
            if has_direct_connection(u, to_resource):
                next_status = collect_resource_in(under_status, to_resource)
                cost = dp[(under_status, u)] + get_direct_distance_between(u, to_resource)
                if cost < dp[(next_status, to_resource)]:
                    dp[(next_status, to_resource)] = cost
                    parent_state[(next_status, to_resource)] = (under_status, u)

    for mask in range(get_total_status_combinations()):
        for u in range(K):
            if dp[(mask, u)] == float('inf'):
                continue
            for v in range(K):
                try_transition_from_resource(u, to_resource=v, under_status=mask)

    # 第三阶段：最优解仲裁
    # 业务意图：从所有 DP 状态计算结果中，查询获得资源分最高、步数最短的那个最优 TSP 状态
    def query_best_tsp_state() -> Tuple[int, int] | None:
        candidates = [
            (mask, u)
            for mask in range(get_total_status_combinations())
            for u in range(K)
            if dp[(mask, u)] != float('inf')
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda state: (evaluate_accumulated_gain_from_bitmap(state[0]), -dp[state])
        )

    best_state = query_best_tsp_state()
    if best_state is None:
        return ResourcePlan(0, [], [], [], {}, [])

    max_gain = evaluate_accumulated_gain_from_bitmap(best_state[0])

    # 业务意图：根据最优终点 TSP 状态，逆向回溯出所经过的资源点索引序列
    def reconstruct_resource_sequence_from(start_state: Tuple[int, int]) -> List[int]:
        sequence: List[int] = []
        curr: Tuple[int, int] | None = start_state
        while curr is not None:
            _, u = curr
            sequence.append(u)
            curr = parent_state.get(curr)
        sequence.reverse()
        return sequence

    resource_sequence = reconstruct_resource_sequence_from(best_state)

    # 业务意图：根据资源点索引序列，拼接出完整的物理路径坐标序列
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
        "dp_states": (1 << K) * K,
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


