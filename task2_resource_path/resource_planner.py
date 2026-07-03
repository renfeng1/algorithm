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
    import heapq

    start = maze.find_unique(Cell.START.value)
    exit_pos = maze.find_unique(Cell.EXIT.value)
    main_path = shortest_path(maze, start, exit_pos)

    registry = MazeResourceRegistry(maze)

    # 评估当前资源收集状态的总净收益（吃金币加分，踩陷阱扣分）
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
        """从高维搜索状态中，提取出当前已收集资源的位图表示"""
        return state[2]

    # 当移动到新坐标时，尝试触发并收集该位置 of 资源（若存在）
    # 技术实现：若坐标命中资源索引表，则将 bitmap 中对应的二进制位设为 1，并返回更新后的 bitmap
    def try_trigger_resource_at(current_bitmap: int, pos: Position) -> int:
        idx = registry.get_resource_id_at(pos)
        if idx is not None:
            return current_bitmap | (1 << idx)
        return current_bitmap

    initial_collected_bitmap = try_trigger_resource_at(0, start)

    # 预先筛选出所有属于金币的资源索引及其分值，以便快速计算乐观上限
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

    # 评估一条路径的资源收集效率分数（获得的资源 / 走的步数）
    # 特殊处理：对于负收益（踩陷阱），步数越多应该越劣（因此乘以 steps）；对于步数为 0 的情况进行安全规避
    def evaluate_efficiency_score(gain: int, steps: int) -> float:
        if steps == 0:
            return float(gain)
        return gain / steps if gain >= 0 else gain * steps

    # =========================================================================
    # 分支限界与剪枝优化设计说明 (Branch and Bound & Pruning Optimization):
    #
    # 1. 优先级队列 (Dijkstra/A* 搜索):
    #    我们使用 heapq 构建优先级队列，根据 "乐观估算效率分数" 进行状态排序，优先拓展潜力最大的状态。
    #    乐观估算效率分数由 "当前得分 + 剩余金币总和"（乐观上限得分）与 "当前步数 + 到终点的曼哈顿距离"（乐观下限步数）计算得出。
    #
    # 2. 分支限界剪枝 (Branch and Bound Pruning):
    #    在搜索过程中，我们维护当前已成功到达终点的 "最优效率分数" (best_exit_efficiency)。
    #    对于任何新状态，如果其 "乐观估算效率分数" 依然小于或等于 best_exit_efficiency，说明该分支绝无可能超越当前最优解，直接剪枝。
    #
    # 3. Beam Search 状态剪枝 (Beam Search State Pruning):
    #    为了应对状态空间爆炸（2^K 规模），我们对每个网格位置限制保留的状态数量（束宽 Beam Width = 128）。
    #    若到达某一位置的候选状态超出 Beam Width 个，我们只保留收益最高/效率最高的 top-128 个状态，其余状态直接舍弃。
    # =========================================================================

    BEAM_WIDTH = 128

    # 初始状态
    initial_state = (start[0], start[1], initial_collected_bitmap)
    initial_score = evaluate_accumulated_gain_from_bitmap(initial_collected_bitmap)
    opt_gain = initial_score + get_optimistic_remaining_gain(initial_collected_bitmap)
    min_remaining_steps = get_manhattan_distance(start, exit_pos)
    opt_efficiency = evaluate_efficiency_score(opt_gain, min_remaining_steps)

    # heap 元素: (优先级(小顶堆存负值), 步数, -当前得分, 行, 列, 资源位图)
    heap = [(-opt_efficiency, 0, -initial_score, start[0], start[1], initial_collected_bitmap)]
    
    # 状态转移父节点表，用于后续逆向回溯重建最优物理路径
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int] | None] = {initial_state: None}
    
    # 记录每个状态的移动步数（路径长度）
    depth: Dict[Tuple[int, int, int], int] = {initial_state: 0}

    # 记录每个网格位置已访问过的状态的得分，用于 Beam Search 剪枝
    cell_beams: Dict[Position, List[int]] = {start: [initial_score]}

    best_exit_efficiency = -float('inf')
    best_exit_state = None

    while heap:
        neg_opt_eff, d, neg_score, r, c, collected_bitmap = heapq.heappop(heap)
        current_state = (r, c, collected_bitmap)
        current_score = -neg_score

        # 如果当前状态到达了终点坐标，更新全局最优终点状态
        if (r, c) == exit_pos:
            actual_eff = evaluate_efficiency_score(current_score, d)
            if actual_eff > best_exit_efficiency:
                best_exit_efficiency = actual_eff
                best_exit_state = current_state
            continue

        # 分支限界：如果当前已找到的终点最优效率高于当前状态的乐观估算值，停止该分支的搜索
        opt_gain = current_score + get_optimistic_remaining_gain(collected_bitmap)
        min_remaining_steps = get_manhattan_distance((r, c), exit_pos)
        opt_efficiency = evaluate_efficiency_score(opt_gain, d + min_remaining_steps)
        if opt_efficiency <= best_exit_efficiency:
            continue

        # 遍历当前位置的相邻网格
        for nxt in maze.neighbors((r, c)):
            next_collected_bitmap = try_trigger_resource_at(collected_bitmap, nxt)
            next_state = (nxt[0], nxt[1], next_collected_bitmap)
            next_depth = d + 1
            next_score = evaluate_accumulated_gain_from_bitmap(next_collected_bitmap)

            # 1. 状态去重与路径松弛：如果曾经以更少或相等的步数到达过此状态，则剪枝
            if next_state in depth and next_depth >= depth[next_state]:
                continue

            # 2. Beam Search 状态剪枝：限制每个网格位置保留的候选状态数
            beams = cell_beams.setdefault(nxt, [])
            if len(beams) >= BEAM_WIDTH and next_score <= beams[-1]:
                continue
            
            # 3. 分支限界剪枝：估算乐观上限效率，若不及当前终点最优，则剪枝
            next_opt_gain = next_score + get_optimistic_remaining_gain(next_collected_bitmap)
            next_min_steps = get_manhattan_distance(nxt, exit_pos)
            next_opt_eff = evaluate_efficiency_score(next_opt_gain, next_depth + next_min_steps)
            if next_opt_eff <= best_exit_efficiency:
                continue

            # 更新 Beam 队列
            beams.append(next_score)
            beams.sort(reverse=True)
            if len(beams) > BEAM_WIDTH:
                beams.pop()

            # 记录新状态的父节点和步数，并推入优先级队列
            parent[next_state] = current_state
            depth[next_state] = next_depth
            heapq.heappush(heap, (-next_opt_eff, next_depth, -next_score, nxt[0], nxt[1], next_collected_bitmap))

    if best_exit_state is None:
        raise ValueError(f"No path from {start} to {exit_pos}")

    # 根据最优终点状态，沿着状态转移链逆向回溯，重建出顺向的物理网格坐标路径
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
        # 在当前收集状态中加入指定资源点，并返回更新后的状态数值表示
        return status | (1 << resource_idx)

    def get_status_space_upper_bound() -> int:
        # 获取收集状态空间的取值上限（2 的 K 次方，用于迭代范围）
        return 1 << K

    def get_shortest_steps_to_state(status: int, end_resource: int) -> float:
        # 查询在指定收集状态且停在特定资源点时的最少累计步数
        return dp[(status, end_resource)]

    def is_shorter_path_to_state(status: int, end_resource: int, cost: float) -> bool:
        # 判定候选路径步数是否比当前已记录的更短
        return cost < dp[(status, end_resource)]

    def record_better_path_to_state(status: int, end_resource: int, cost: float, predecessor: Tuple[int, int] | None) -> None:
        # 更新记录更优的路径步数，并建立前驱状态连接以供回溯
        dp[(status, end_resource)] = cost
        if predecessor is not None:
            parent_state[(status, end_resource)] = predecessor

    def get_predecessor_of_state(state: Tuple[int, int]) -> Tuple[int, int] | None:
        # 查询指定 TSP 状态的前驱父状态
        return parent_state.get(state)

    # 第二阶段：TSP 状态压缩 DP 结合分支限界与 Beam Search
    def filter_positive_reward_resources() -> Tuple[List[int], List[int]]:
        """从注册表中筛选所有能提供正向收益的资源，返回其 ID 集合与对应分值。"""
        indices = [
            i for i in range(K)
            if registry.get_resource_value(i) > 0
        ]
        values = [registry.get_resource_value(i) for i in indices]
        return indices, values

    gold_indices, gold_values = filter_positive_reward_resources()

    def get_optimistic_remaining_gain(status: int) -> int:
        #乐观：如果全部都被收集得到的最优解
        #假定后续收集这些未触及的金币时，不需要花费任何移动步数开销
        #假定在前往这些金币的途中，不需要踩到任何扣分的陷阱格
        remaining_gain = 0
        for i, val in zip(gold_indices, gold_values):
            if not (status & (1 << i)):
                remaining_gain += val
        return remaining_gain

    # DP 状态表，记录 (status, endpoint) -> 最少步数
    dp: Dict[Tuple[int, int], float] = {}
    #记录状态转移的“来路”（即前驱状态），以便在算法最终找到最优的 TSP 终点状态后，能够逆向回溯出一条完整的物理收集路径。
    parent_state: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def record_shortest_steps(state: Tuple[int, int], steps: float) -> None:
        """记录到达特定资源收集状态的最短移动步数"""
        dp[state] = steps

    def record_predecessor(state: Tuple[int, int], predecessor: Tuple[int, int]) -> None:
        """记录状态转移的前驱节点，用于最优路径回溯"""
        parent_state[state] = predecessor

    class LevelStatePool:
        """用于管理和查询同一层级（收集了相同数量资源）的所有搜索状态"""
        def __init__(self):
            self._states: Dict[Tuple[int, int], float] = {}
        
        def add_or_relax_state(self, state: Tuple[int, int], cost: float) -> bool:
            """尝试添加新状态，如果到达该状态的步数更短，则状态松弛更新并返回 True"""
            if cost < self._states.get(state, float('inf')):
                self._states[state] = cost
                return True
            return False
            
        def get_all_states(self):
            """查询当前层级下的所有存活状态及其花费步数"""
            return self._states.items()
            
        def keep_top_k_candidates(self, beam_width: int) -> None:
            """Beam Search 状态剪枝：仅保留收益最高、步数最少的前 beam_width 个状态"""
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

    # 初始状态：只收集了单个资源点的状态，步数为 0
    current_level_pool = LevelStatePool()
    for i in range(K):
        state = (initial_status_of(i), i)
        current_level_pool.add_or_relax_state(state, 0.0)
        record_shortest_steps(state, 0.0)

    # 跟踪当前的全局最优解 (最大收集收益，最小步数)
    historical_max_gain = -float('inf')
    historical_min_cost = float('inf')
    historical_best_state = None

    def update_global_best(status: int, endpoint: int, cost: float) -> None:
        #如果最佳则更新，不最佳不更新
        nonlocal historical_max_gain, historical_min_cost, historical_best_state
        gain = evaluate_accumulated_gain_from_status(status)
        if (gain > historical_max_gain) or (gain == historical_max_gain and cost < historical_min_cost):
            historical_max_gain = gain
            historical_min_cost = cost
            historical_best_state = (status, endpoint)
    #玩家可以“空降”（直接出现）在地图上的任意一个资源点上作为起点，并且不需要走到特定的终点。
    for state, cost in current_level_pool.get_all_states():
        update_global_best(state[0], state[1], cost)#初始化作为起点为更新

    # 束宽设置，防止当 K 较大时状态数呈指数级爆炸
    #这个值太小会影响正确性，理论上有丢弃最优解的风险，但在迷宫资源的场景下，我们的排序规则（优先总收益，其次短步数）极其契合实际
    BEAM_WIDTH = 256

    def should_prune_by_branch_and_bound(status: int, cost: float) -> bool:
        """分支限界剪枝：如果当前状态的乐观估算收益不及已知的全局最优解，则判定为无望分支进行剪枝"""
        opt_gain = evaluate_accumulated_gain_from_status(status) + get_optimistic_remaining_gain(status)
        if opt_gain < historical_max_gain:
            return True
        if opt_gain == historical_max_gain and cost >= historical_min_cost:
            return True
        return False

    # 逐层扩展（level 表示已收集的资源点个数，从 1 到 K-1）
    for level in range(1, K):
        next_level_pool = LevelStatePool()

        for (status, endpoint), cost in current_level_pool.get_all_states():
            # 1. 分支限界剪枝：基于乐观估算判定是否属于无望分支
            if should_prune_by_branch_and_bound(status, cost):
                continue

            for next_resource in range(K):
                if not is_resource_collected_in_status(status, next_resource):
                    if has_direct_connection(endpoint, next_resource):
                        next_status = collect_resource_in_status(status, next_resource)
                        next_cost = cost + get_direct_distance_between(endpoint, next_resource)
                        next_state = (next_status, next_resource)

                        # 2. 状态松弛：如果找到到达 (next_status, next_resource) 步数更少的路径，则更新
                        if next_level_pool.add_or_relax_state(next_state, next_cost):
                            record_shortest_steps(next_state, next_cost)
                            record_predecessor(next_state, (status, endpoint))
                            update_global_best(next_status, next_resource, next_cost)

        # 3. Beam Search 状态剪枝
        next_level_pool.keep_top_k_candidates(BEAM_WIDTH)
        current_level_pool = next_level_pool

        if current_level_pool.is_empty():
            break

    best_state = historical_best_state
    if best_state is None:
        return ResourcePlan(0, [], [], [], {}, [])

    max_gain = evaluate_accumulated_gain_from_status(best_state[0])






    # 根据最优终点 TSP 状态，逆向回溯出所经过的资源点索引序列
    def reconstruct_resource_sequence_from(start_state: Tuple[int, int]) -> List[int]:
        sequence: List[int] = []
        curr: Tuple[int, int] | None = start_state
        while curr is not None:
            _, endpoint = curr
            sequence.append(endpoint)
            curr = get_predecessor_of_state(curr)
        sequence.reverse()
        return sequence

    resource_sequence = reconstruct_resource_sequence_from(best_state)

    # 根据资源点索引序列，拼接出完整的物理路径坐标序列
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
        "dp_states": get_status_space_upper_bound() * K,
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


