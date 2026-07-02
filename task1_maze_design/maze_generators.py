from __future__ import annotations

import heapq
import random
from collections import deque
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from common.models import Cell, GeneratedMaze


Room = Tuple[int, int]
Edge = Tuple[Room, Room]


def _grid_template(size: int) -> List[List[str]]:
    return [[Cell.WALL.value for _ in range(size)] for _ in range(size)]


def _room_to_cell(room: Room) -> Tuple[int, int]:
    r, c = room
    return 2 * r + 1, 2 * c + 1


def _carve_room(grid: List[List[str]], room: Room) -> None:
    r, c = _room_to_cell(room)
    grid[r][c] = Cell.ROAD.value


def _carve_edge(grid: List[List[str]], a: Room, b: Room) -> None:
    ar, ac = _room_to_cell(a)
    br, bc = _room_to_cell(b)
    grid[(ar + br) // 2][(ac + bc) // 2] = Cell.ROAD.value
    grid[ar][ac] = Cell.ROAD.value
    grid[br][bc] = Cell.ROAD.value


def _neighbors(room: Room, rows: int, cols: int) -> Iterable[Room]:
    r, c = room
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def _all_rooms(rows: int, cols: int) -> List[Room]:
    return [(r, c) for r in range(rows) for c in range(cols)]


def _snapshot_from_edges(
    size: int,
    rows: int,
    cols: int,
    edges: Sequence[Edge],
) -> List[List[str]]:
    grid = _grid_template(size)
    for room in _all_rooms(rows, cols):
        _carve_room(grid, room)
    for a, b in edges:
        _carve_edge(grid, a, b)
    return grid


def _build_generated(
    size: int,
    rows: int,
    cols: int,
    edges: List[Edge],
    snapshots_edges: List[List[Edge]],
    events: List[dict],
    generator_name: str,
    seed: int,
) -> GeneratedMaze:
    final_grid = _snapshot_from_edges(size, rows, cols, edges)
    snapshots = [_snapshot_from_edges(size, rows, cols, item) for item in snapshots_edges]
    return GeneratedMaze(
        grid=final_grid,
        room_rows=rows,
        room_cols=cols,
        edges=edges[:],
        snapshots=snapshots,
        events=events[:],
        generator_name=generator_name,
        random_seed=seed,
    )


def generate_divide_conquer(size: int, seed: int) -> GeneratedMaze:
    rng = random.Random(seed)
    rows = cols = (size - 1) // 2
    edges: List[Edge] = []
    snapshots: List[List[Edge]] = []
    events: List[dict] = []

    def connect_subgrid(r0: int, r1: int, c0: int, c1: int) -> Room:
        if r0 == r1 and c0 == c1:
            return (r0, c0)
        if (r1 - r0) >= (c1 - c0):
            mid = (r0 + r1) // 2
            top = connect_subgrid(r0, mid, c0, c1)
            bottom = connect_subgrid(mid + 1, r1, c0, c1)
            join_c = rng.randint(c0, c1)
            a = (mid, join_c)
            b = (mid + 1, join_c)
            edges.append((a, b))
            snapshots.append(edges[:])
            events.append(
                {
                    "step": len(edges),
                    "action": "connect",
                    "from_room": list(a),
                    "to_room": list(b),
                    "strategy": "vertical_merge",
                    "region": [r0, r1, c0, c1],
                }
            )
            return rng.choice([top, bottom])
        mid = (c0 + c1) // 2
        left = connect_subgrid(r0, r1, c0, mid)
        right = connect_subgrid(r0, r1, mid + 1, c1)
        join_r = rng.randint(r0, r1)
        a = (join_r, mid)
        b = (join_r, mid + 1)
        edges.append((a, b))
        snapshots.append(edges[:])
        events.append(
            {
                "step": len(edges),
                "action": "connect",
                "from_room": list(a),
                "to_room": list(b),
                "strategy": "horizontal_merge",
                "region": [r0, r1, c0, c1],
            }
        )
        return rng.choice([left, right])

    connect_subgrid(0, rows - 1, 0, cols - 1)
    return _build_generated(size, rows, cols, edges, snapshots, events, "divide_conquer", seed)


def generate_dfs_backtracking(size: int, seed: int) -> GeneratedMaze:
    rng = random.Random(seed)
    rows = cols = (size - 1) // 2
    start = (0, 0)
    stack = [start]
    visited: Set[Room] = {start}
    edges: List[Edge] = []
    snapshots: List[List[Edge]] = []
    events: List[dict] = []

    while stack:
        current = stack[-1]
        choices = [n for n in _neighbors(current, rows, cols) if n not in visited]
        if not choices:
            stack.pop()
            continue
        nxt = rng.choice(choices)
        visited.add(nxt)
        stack.append(nxt)
        edges.append((current, nxt))
        snapshots.append(edges[:])
        events.append(
            {
                "step": len(edges),
                "action": "connect",
                "from_room": list(current),
                "to_room": list(nxt),
                "strategy": "dfs_expand",
                "frontier_size": len(stack),
            }
        )

    return _build_generated(size, rows, cols, edges, snapshots, events, "dfs_backtracking", seed)


def generate_bfs_expansion(size: int, seed: int) -> GeneratedMaze:
    rng = random.Random(seed)
    rows = cols = (size - 1) // 2
    start = (0, 0)
    queue = deque([start])
    visited: Set[Room] = {start}
    edges: List[Edge] = []
    snapshots: List[List[Edge]] = []
    events: List[dict] = []

    while queue:
        current = queue.popleft()
        choices = [n for n in _neighbors(current, rows, cols) if n not in visited]
        rng.shuffle(choices)
        for nxt in choices:
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)
            edges.append((current, nxt))
            snapshots.append(edges[:])
            events.append(
                {
                    "step": len(edges),
                    "action": "connect",
                    "from_room": list(current),
                    "to_room": list(nxt),
                    "strategy": "bfs_expand",
                    "frontier_size": len(queue),
                }
            )

    return _build_generated(size, rows, cols, edges, snapshots, events, "bfs_expansion", seed)


def generate_mst_greedy(size: int, seed: int) -> GeneratedMaze:
    rng = random.Random(seed)
    rows = cols = (size - 1) // 2
    start = (0, 0)
    visited: Set[Room] = {start}
    heap: List[Tuple[int, Room, Room]] = []
    edges: List[Edge] = []
    snapshots: List[List[Edge]] = []
    events: List[dict] = []

    def push_from(room: Room) -> None:
        for nxt in _neighbors(room, rows, cols):
            if nxt not in visited:
                heapq.heappush(heap, (rng.randint(1, 10_000), room, nxt))

    push_from(start)
    while heap and len(visited) < rows * cols:
        _, a, b = heapq.heappop(heap)
        if b in visited:
            continue
        visited.add(b)
        edges.append((a, b))
        snapshots.append(edges[:])
        events.append(
            {
                "step": len(edges),
                "action": "connect",
                "from_room": list(a),
                "to_room": list(b),
                "strategy": "mst_pick_min_edge",
                "heap_size": len(heap),
            }
        )
        push_from(b)

    return _build_generated(size, rows, cols, edges, snapshots, events, "mst_greedy", seed)


def generate_all(size: int, seed: int) -> List[GeneratedMaze]:
    return [generate_named(size=size, seed=seed, name=name) for name in generator_names()]


def generator_names() -> List[str]:
    return ["divide_conquer", "mst_greedy", "dfs_backtracking", "bfs_expansion"]


def generate_named(size: int, seed: int, name: str) -> GeneratedMaze:
    offset_map = {
        "divide_conquer": 11,
        "mst_greedy": 23,
        "dfs_backtracking": 37,
        "bfs_expansion": 53,
    }
    if name not in offset_map:
        raise ValueError(f"Unknown generator: {name}")
    real_seed = seed + offset_map[name]
    if name == "divide_conquer":
        return generate_divide_conquer(size=size, seed=real_seed)
    if name == "mst_greedy":
        return generate_mst_greedy(size=size, seed=real_seed)
    if name == "dfs_backtracking":
        return generate_dfs_backtracking(size=size, seed=real_seed)
    return generate_bfs_expansion(size=size, seed=real_seed)


