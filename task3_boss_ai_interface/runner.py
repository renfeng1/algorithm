from __future__ import annotations

from pathlib import Path

from common.io_utils import save_json
from task3_boss_ai_interface.adapters import (
    task2_dp_result,
    task3_boss_result,
    task4_resource_pickup_result,
    task5_explore_result,
)


def run_interface_task(task: str, input_path: str, output_path: str | None = None) -> dict:
    if task == "task2":
        result = task2_dp_result(input_path)
    elif task == "task3":
        result = task3_boss_result(input_path)
    elif task == "task4":
        result = task4_resource_pickup_result(input_path)
    elif task == "task5":
        result = task5_explore_result(input_path)
    else:
        raise ValueError(f"Unknown interface task: {task}")

    if output_path:
        save_json(output_path, result)
    return result


