from __future__ import annotations

from common.models import Cell


GOLD_VALUE = 50
TRAP_VALUE = -30
VISION_RADIUS = 1


def cell_resource_delta(cell: str) -> int:
    if cell == Cell.GOLD.value:
        return GOLD_VALUE
    if cell == Cell.TRAP.value:
        return TRAP_VALUE
    return 0

