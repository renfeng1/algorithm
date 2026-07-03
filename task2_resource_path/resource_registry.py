from typing import Dict, List
from common.models import Cell, MazeGame, Position
from common.rules import GOLD_VALUE, TRAP_VALUE

class MazeResourceRegistry:
    """
    业务意图：统一管理和查询迷宫内金币和陷阱等资源的注册表。
    通过业务概念（如资源 ID、位置、分值）隐藏底层的状态压缩位图实现，表达 "what" 与 "why"。
    """
    def __init__(self, maze: MazeGame):
        self.maze = maze
        self.all_resource_positions = [
            (r, c)
            for r, row in enumerate(maze.grid)
            for c, value in enumerate(row)
            if value in {Cell.GOLD.value, Cell.TRAP.value}
        ]
        self._position_to_id = {pos: i for i, pos in enumerate(self.all_resource_positions)}
        self._id_to_value = [
            GOLD_VALUE if maze.grid[r][c] == Cell.GOLD.value else TRAP_VALUE
            for r, c in self.all_resource_positions
        ]

    def get_resource_id_at(self, pos: Position) -> int | None:
        """业务意图：获取网格坐标处资源的唯一标识符。如果该坐标没有资源，返回 None。"""
        return self._position_to_id.get(pos)

    def get_resource_value(self, resource_id: int) -> int:
        """业务意图：获取指定资源 ID 对应的业务价值（金币为正分，陷阱为负分）。"""
        return self._id_to_value[resource_id]

    def get_resource_position(self, resource_id: int) -> Position:
        """业务意图：获取指定资源 ID 对应的物理网格坐标。"""
        return self.all_resource_positions[resource_id]

    def get_total_resources_count(self) -> int:
        """业务意图：获取地图上所有注册的可触发资源的总数。"""
        return len(self.all_resource_positions)

    def has_resource_at(self, pos: Position) -> bool:
        """业务意图：判断指定的网格坐标上是否注册了资源。"""
        return pos in self._position_to_id
