from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from task1_maze_design.map_designer import design_maze
from task1_maze_design.maze_generators import generate_all, generate_named, generator_names
from task2_resource_path.resource_planner import plan_optimal_resource_path
from common.models import MazeConfig
from visualization.interactive_bundle import _serialize_generated, _serialize_step_frames, _viewer_html


def _normalize_size(size: int) -> int:
    """Maze generators use odd n so rooms sit on odd grid coordinates."""
    try:
        value = int(size)
    except (TypeError, ValueError):
        value = 15
    value = max(5, value)
    if value % 2 == 0:
        value += 1
    return value


def _normalize_seed(seed: int) -> int:
    try:
        return max(0, int(seed))
    except (TypeError, ValueError):
        return 42


def _best_resource_for_generated(generated_list, config: MazeConfig):
    candidates = []
    for generated in generated_list:
        maze, metrics = design_maze(generated, config)
        resource = plan_optimal_resource_path(maze)
        score = maze.design_score + resource.max_resource * 0.9
        candidates.append((score, generated, maze, resource, metrics))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0]


def _algorithm_payload_with_design(generated, config: MazeConfig) -> tuple[dict, object, object, object]:
    maze, metrics = design_maze(generated, config)
    resource = plan_optimal_resource_path(maze)
    payload = _serialize_generated(
        generated,
        decorated_grid=maze.grid,
        decorated_description=(
            f"{generated.generator_name}: 迷宫拓扑生成完成，随后按地图设计规则放置 S/E/B/G/T，"
            "该最终帧与接口输出迷宫语义一致。"
        ),
    )
    return payload, maze, resource, metrics


def _live_payload(size: int, seed: int) -> dict:
    size = _normalize_size(size)
    seed = _normalize_seed(seed)
    config = MazeConfig(size=size, random_seed=seed)
    generated_list = generate_all(size=size, seed=seed)
    algorithm_payloads = []
    candidates = []
    for generated in generated_list:
        payload, maze, resource, metrics = _algorithm_payload_with_design(generated, config)
        algorithm_payloads.append(payload)
        score = maze.design_score + resource.max_resource * 0.9
        candidates.append((score, generated, maze, resource, metrics))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_generated, maze, resource, _ = candidates[0]
    return {
        "size": size,
        "seed": seed,
        "selection_rule": "同一个 n 和 seed 下同时生成四种算法候选，按地图设计分 + 0.9 * 最优资源收益选择路径规划展示图。",
        "algorithms": algorithm_payloads,
        "resource_process": {
            "name": "best_resource_path",
            "source_algorithm": best_generated.generator_name,
            "final_grid": [row[:] for row in maze.grid],
            "frames": _serialize_step_frames(resource.frames),
        },
        "available_algorithms": generator_names(),
    }


def _single_algorithm_payload(size: int, seed: int, algorithm: str) -> dict:
    size = _normalize_size(size)
    seed = _normalize_seed(seed)
    if algorithm not in generator_names():
        raise ValueError(f"Unknown generator: {algorithm}")
    config = MazeConfig(size=size, random_seed=seed)
    generated = generate_named(size=size, seed=seed, name=algorithm)
    algorithm_payload, maze, resource, _ = _algorithm_payload_with_design(generated, config)
    return {
        "size": size,
        "seed": seed,
        "selection_rule": "只生成当前选中的算法，并在该算法生成出的地图上做最佳资源路径规划。",
        "algorithms": [algorithm_payload],
        "resource_process": {
            "name": "best_resource_path",
            "source_algorithm": algorithm,
            "final_grid": [row[:] for row in maze.grid],
            "frames": _serialize_step_frames(resource.frames),
        },
        "available_algorithms": generator_names(),
    }


def _live_viewer_html() -> str:
    return _viewer_html(live=True)


class _Handler(BaseHTTPRequestHandler):
    server_version = "MazeLiveViewer/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/viewer.html"}:
            self._send_html(_live_viewer_html())
            return

        if parsed.path == "/api/bundle":
            query = parse_qs(parsed.query)
            size = int(query.get("n", ["15"])[0])
            seed = int(query.get("seed", ["42"])[0])
            payload = _live_payload(size=size, seed=seed)
            self._send_json(payload)
            return

        if parsed.path == "/api/single":
            query = parse_qs(parsed.query)
            size = int(query.get("n", ["15"])[0])
            seed = int(query.get("seed", ["42"])[0])
            algorithm = query.get("algorithm", ["bfs_expansion"])[0]
            try:
                payload = _single_algorithm_payload(size=size, seed=seed, algorithm=algorithm)
            except ValueError as exc:
                self._send_json({"error": str(exc), "available_algorithms": generator_names()}, status=400)
                return
            self._send_json(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return

    def _send_html(self, text: str):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_live_viewer(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> str:
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/viewer.html"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        print(url)
        server.serve_forever()
    finally:
        server.server_close()
    return url


