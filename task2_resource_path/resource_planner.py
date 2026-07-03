from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

from common.models import Cell, MazeGame, Position, StepFrame
from common.pathing import shortest_path
from common.rules import GOLD_VALUE, TRAP_VALUE


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


class MazeResourceRegistry:
    """
    业务意图：统一管理和查询迷宫内金币和陷阱等资源的注册表。
    通过查询函数替代原有的临时变量，以更好地体现各个数据结构的业务含义。
    """
    def __init__(self, maze: MazeGame):
        self.maze = maze
        self.all_resource_positions = [
            (r, c)
            for r, row in enumerate(maze.grid)
            for c, value in enumerate(row)
            if value in {Cell.GOLD.value, Cell.TRAP.value}
        ]
        self._position_to_bitmap_index = {pos: i for i, pos in enumerate(self.all_resource_positions)}
        self._bitmap_index_to_value = [
            GOLD_VALUE if maze.grid[r][c] == Cell.GOLD.value else TRAP_VALUE
            for r, c in self.all_resource_positions
        ]

    def get_bitmap_index(self, pos: Position) -> int | None:
        """根据网格坐标，查询其在收集状态位图（collected_bitmap）中对应的二进制位索引"""
        return self._position_to_bitmap_index.get(pos)

    def get_value_by_bitmap_index(self, index: int) -> int:
        """根据位图中的二进制位索引，查询该资源的净收益值（金币为正，陷阱为负）"""
        return self._bitmap_index_to_value[index]

    def get_resource_position_by_index(self, index: int) -> Position:
        """根据二进制位索引，获取该资源的物理网格坐标"""
        return self.all_resource_positions[index]

    def total_resources_count(self) -> int:
        """获取地图上所有可触发资源（金币和陷阱）的总数"""
        return len(self.all_resource_positions)

    def contains_resource(self, pos: Position) -> bool:
        """判断某个物理网格坐标是否是注册的资源格"""
        return pos in self._position_to_bitmap_index


def plan_optimal_resource_path(maze: MazeGame) -> ResourcePlan:
    start = maze.find_unique(Cell.START.value)
    exit_pos = maze.find_unique(Cell.EXIT.value)
    main_path = shortest_path(maze, start, exit_pos)

    registry = MazeResourceRegistry(maze)

    # 业务意图：评估当前资源收集状态的总净收益（吃金币加分，踩陷阱扣分）
    # 技术实现：利用 lowbit (Brian Kernighan 算法) 极速跳过 bitmap 中的 0，仅对为 1 的资源位进行累加
    def evaluate_accumulated_gain(collected_bitmap: int) -> int:
        total = 0
        current = collected_bitmap
        while current:
            bit = current & -current
            total += registry.get_value_by_bitmap_index(bit.bit_length() - 1)
            current -= bit
        return total

    # 业务意图：当移动到新坐标时，尝试触发并收集该位置的资源（若存在）
    # 技术实现：若坐标命中资源索引表，则将 bitmap 中对应的二进制位设为 1，并返回更新后的 bitmap
    def try_trigger_resource_at(current_bitmap: int, pos: Position) -> int:
        idx = registry.get_bitmap_index(pos)
        if idx is not None:
            return current_bitmap | (1 << idx)
        return current_bitmap

    initial_collected_bitmap = try_trigger_resource_at(0, start)

    state = (start[0], start[1], initial_collected_bitmap)
    queue = deque([state])
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int] | None] = {state: None}
    depth: Dict[Tuple[int, int, int], int] = {state: 0}
    best_exit_state = state if start == exit_pos else None

    while queue:
        r, c, collected_bitmap = queue.popleft()
        current_state = (r, c, collected_bitmap)
        if (r, c) == exit_pos:
            if best_exit_state is None:
                best_exit_state = current_state
            else:
                current_key = (evaluate_accumulated_gain(collected_bitmap), -depth[current_state])
                best_key = (evaluate_accumulated_gain(best_exit_state[2]), -depth[best_exit_state])
                if current_key > best_key:
                    best_exit_state = current_state
            continue

        for nxt in maze.neighbors((r, c)):
            next_collected_bitmap = try_trigger_resource_at(collected_bitmap, nxt)
            next_state = (nxt[0], nxt[1], next_collected_bitmap)
            if next_state in parent:
                continue
            parent[next_state] = current_state
            depth[next_state] = depth[current_state] + 1
            queue.append(next_state)

    if best_exit_state is None:
        raise ValueError(f"No path from {start} to {exit_pos}")

    walk_path: List[Position] = []
    cursor: Tuple[int, int, int] | None = best_exit_state
    while cursor is not None:
        walk_path.append((cursor[0], cursor[1]))
        cursor = parent[cursor]
    walk_path.reverse()

    max_resource = evaluate_accumulated_gain(best_exit_state[2])
    branch_gains: Dict[str, int] = {
        "resource_cells": registry.total_resources_count(),
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
        idx = registry.get_bitmap_index(pos)
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
    K = registry.total_resources_count()
    
    if K == 0:
        return ResourcePlan(0, [], [], [], {}, [])

    def evaluate_accumulated_gain(collected_bitmap: int) -> int:
        total = 0
        current = collected_bitmap
        while current:
            bit = current & -current
            total += registry.get_value_by_bitmap_index(bit.bit_length() - 1)
            current -= bit
        return total

    # 第一阶段：提取资源图 (Graph Abstraction)
    dist = [[float('inf')] * K for _ in range(K)]
    path_between = [[[] for _ in range(K)] for _ in range(K)]

    for i, start_pos in enumerate(registry.all_resource_positions):
        dist[i][i] = 0
        path_between[i][i] = [start_pos]
        queue = deque([start_pos])
        parent_map = {start_pos: None}
        
        while queue:
            curr = queue.popleft()
            
            # 关键：如果在 BFS 过程中遇到了【其他的资源点】，停止从该点继续向外探索。
            # 这样找到的最短路必然是“纯粹”的：要么是不穿过任何其他资源的直达路径，
            # 要么是绕开其他资源的避让路径。穿过其他资源的路径将由后续 DP 自行组合。
            if curr != start_pos and registry.contains_resource(curr):
                continue
                
            for nxt in maze.neighbors(curr):
                if nxt not in parent_map:
                    parent_map[nxt] = curr
                    queue.append(nxt)
                    
        for j, end_pos in enumerate(registry.all_resource_positions):
            if i != j and end_pos in parent_map:
                path = []
                cursor: Tuple[int, int] | None = end_pos
                while cursor is not None:
                    path.append(cursor)
                    cursor = parent_map[cursor]
                path.reverse()
                dist[i][j] = len(path) - 1
                path_between[i][j] = path

    # 第二阶段：TSP 状态压缩 DP
    dp: Dict[Tuple[int, int], float] = { (mask, u): float('inf') for mask in range(1 << K) for u in range(K) }
    parent_state: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # 初始状态：可以任意选择一个资源点空降开局，步数为 0
    for i in range(K):
        dp[(1 << i, i)] = 0

    for mask in range(1 << K):
        for u in range(K):
            if dp[(mask, u)] == float('inf'):
                continue
            for v in range(K):
                if not (mask & (1 << v)):
                    if dist[u][v] != float('inf'):
                        next_mask = mask | (1 << v)
                        cost = dp[(mask, u)] + dist[u][v]
                        if cost < dp[(next_mask, v)]:
                            dp[(next_mask, v)] = cost
                            parent_state[(next_mask, v)] = (mask, u)

    # 第三阶段：最优解仲裁
    # 默认最优是什么都不做（收益 0，步数 0）
    max_gain = 0
    min_steps = 0
    best_state = None

    for mask in range(1 << K):
        gain = evaluate_accumulated_gain(mask)
        for u in range(K):
            steps = dp[(mask, u)]
            if steps != float('inf'):
                if gain > max_gain or (gain == max_gain and steps < min_steps):
                    max_gain = gain
                    min_steps = steps
                    best_state = (mask, u)

    if best_state is None:
        return ResourcePlan(0, [], [], [], {}, [])

    # 回溯资源点序列
    resource_sequence = []
    curr_state: Tuple[int, int] | None = best_state
    while curr_state is not None:
        mask, u = curr_state
        resource_sequence.append(u)
        curr_state = parent_state.get(curr_state)
    resource_sequence.reverse()

    # 拼接物理路径
    walk_path: List[Position] = []
    if len(resource_sequence) == 1:
        u = resource_sequence[0]
        walk_path = [registry.get_resource_position_by_index(u)]
    else:
        for idx in range(len(resource_sequence) - 1):
            u = resource_sequence[idx]
            v = resource_sequence[idx+1]
            segment = path_between[u][v]
            if not walk_path:
                walk_path.extend(segment)
            else:
                walk_path.extend(segment[1:])

    resource_cells_in_order = [registry.get_resource_position_by_index(u) for u in resource_sequence]

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
        if registry.contains_resource(pos) and pos not in triggered:
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


