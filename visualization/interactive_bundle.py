from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.io_utils import ensure_parent, save_json
from common.models import GeneratedMaze, MazeGame, StepFrame


def _serialize_grid(grid):
    return [row[:] for row in grid]


def _serialize_generated(
    generated: GeneratedMaze,
    decorated_grid: Sequence[Sequence[str]] | None = None,
    decorated_description: str | None = None,
) -> dict[str, Any]:
    frames = []
    for index, grid in enumerate(generated.snapshots):
        event = generated.events[index] if index < len(generated.events) else {}
        frames.append(
            {
                "index": index + 1,
                "grid": _serialize_grid(grid),
                "title": f"{generated.generator_name} step {index + 1}",
                "description": _event_description(generated.generator_name, event),
                "event": event,
            }
        )
    if not frames:
        frames.append(
            {
                "index": 1,
                "grid": _serialize_grid(generated.grid),
                "title": f"{generated.generator_name} final",
                "description": "No intermediate frame was recorded.",
                "event": {},
            }
        )
    if decorated_grid is not None:
        frames.append(
            {
                "index": len(frames) + 1,
                "grid": _serialize_grid(decorated_grid),
                "title": f"{generated.generator_name} final designed maze",
                "description": decorated_description
                or "拓扑迷宫生成完成后，地图设计阶段放置起点 S、终点 E、BOSS B、金币 G、陷阱 T。",
                "event": {"action": "decorate_map", "strategy": "map_design"},
            }
        )
    return {
        "name": generated.generator_name,
        "size": len(decorated_grid) if decorated_grid is not None else len(generated.grid),
        "frames": frames,
        "final_grid": _serialize_grid(decorated_grid if decorated_grid is not None else generated.grid),
    }


def _serialize_step_frames(frames: Sequence[StepFrame]) -> list[dict[str, Any]]:
    out = []
    for index, frame in enumerate(frames):
        out.append(
            {
                "index": index + 1,
                "grid": _serialize_grid(frame.grid),
                "title": frame.title,
                "description": frame.description,
                "path": [[r, c] for r, c in frame.path],
                "highlights": [[r, c] for r, c in frame.highlights],
                "meta": frame.meta,
            }
        )
    return out


def _event_description(name: str, event: dict[str, Any]) -> str:
    if not event:
        return f"{name}: generation frame"
    a = tuple(event.get("from_room", []))
    b = tuple(event.get("to_room", []))
    strategy = event.get("strategy", "step")
    return f"{strategy}: connect room {a} -> {b}"


def create_visualization_bundle(
    bundle_dir: str | Path,
    generated_list: Sequence[GeneratedMaze],
    best_maze: MazeGame,
    resource_frames: Sequence[StepFrame],
    auto_open: bool = False,
    decorated_grids: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> Path:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_path = bundle_dir / "viewer_data.json"
    html_path = bundle_dir / "viewer.html"

    decorated_grids = decorated_grids or {}
    payload = {
        "size": len(best_maze.grid),
        "algorithms": [
            _serialize_generated(
                generated,
                decorated_grid=decorated_grids.get(generated.generator_name),
            )
            for generated in generated_list
        ],
        "resource_process": {
            "name": "best_resource_path",
            "final_grid": _serialize_grid(best_maze.grid),
            "frames": _serialize_step_frames(resource_frames),
        },
    }
    save_json(data_path, payload)
    ensure_parent(html_path)
    html_path.write_text(_viewer_html(live=False), encoding="utf-8")
    if auto_open:
        webbrowser.open(html_path.resolve().as_uri())
    return html_path


def _viewer_html(live: bool = False) -> str:
    live_flag = "true" if live else "false"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>迷宫算法可视化实验台</title>
  <style>
    :root {{
      --bg: #efe7d2;
      --ink: #172033;
      --paper: #fffaf0;
      --accent: #c65f21;
      --muted: #667085;
      --line: rgba(23, 32, 51, 0.14);
      --wall: #172033;
      --road: #fffdf7;
      --start: #12805c;
      --exit: #d33d2e;
      --boss: #5146aa;
      --gold: #f2b84b;
      --trap: #2586a7;
      --path: #101828;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Noto Serif SC", "Songti SC", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 8%, rgba(198, 95, 33, 0.20), transparent 28%),
        radial-gradient(circle at 85% 90%, rgba(37, 134, 167, 0.16), transparent 34%),
        linear-gradient(135deg, #f6eedf 0%, var(--bg) 100%);
    }}
    .shell {{
      width: min(1360px, calc(100vw - 28px));
      margin: 18px auto;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 18px;
    }}
    .panel {{
      background: rgba(255, 250, 240, 0.90);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: 0 24px 70px rgba(23, 32, 51, 0.10);
      backdrop-filter: blur(10px);
    }}
    .sidebar {{
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 31px;
      line-height: 1.08;
      letter-spacing: 0.02em;
    }}
    .lead {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
    }}
    .group {{
      display: flex;
      flex-direction: column;
      gap: 9px;
    }}
    .label {{
      font-size: 12px;
      letter-spacing: 0.14em;
      color: var(--muted);
      text-transform: uppercase;
    }}
    select, button, input {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fffdf7;
      color: var(--ink);
      border-radius: 14px;
      padding: 11px 13px;
      font-size: 15px;
    }}
    button {{
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}
    button:hover {{
      transform: translateY(-1px);
      border-color: rgba(198, 95, 33, 0.45);
      background: #fff7e8;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }}
    .live-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .live-grid button {{
      grid-column: span 2;
    }}
    .meta {{
      min-height: 166px;
      padding: 14px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(198, 95, 33, 0.12), rgba(255, 255, 255, 0.28));
    }}
    .meta h2 {{
      margin: 0 0 8px;
      font-size: 19px;
    }}
    .meta p {{
      margin: 0;
      color: #384152;
      line-height: 1.65;
      font-size: 14px;
    }}
    .stage {{
      padding: 18px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
    }}
    .stage-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      padding: 2px 8px 0;
    }}
    .stage-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .stage-head span {{
      color: var(--muted);
      font-size: 14px;
    }}
    .canvas-wrap {{
      min-height: 640px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
      border-radius: 22px;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.74), rgba(239, 231, 210, 0.40)),
        var(--paper);
      overflow: hidden;
    }}
    canvas {{
      width: min(78vh, 100%);
      max-width: 100%;
      height: auto;
      border-radius: 18px;
      background: white;
      box-shadow: inset 0 0 0 1px rgba(23, 32, 51, 0.08);
    }}
    .timeline {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 0 4px;
    }}
    .tiny {{
      font-size: 13px;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .canvas-wrap {{
        min-height: 420px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel sidebar">
      <section>
        <h1>迷宫算法<br />可视化实验台</h1>
        <p class="lead">支持四种生成算法、最佳资源路径过程、播放/暂停/逐步查看。实时模式下可输入 n 重新生成。</p>
      </section>

      <div class="group" id="live-panel">
        <div class="label">实时生成</div>
        <div class="live-grid">
          <input id="size-input" type="number" min="5" step="2" value="15" />
          <input id="seed-input" type="number" min="0" step="1" value="42" />
          <button id="regenerate">按 n 生成四种算法</button>
          <button id="single-regenerate">只生成当前算法</button>
        </div>
      </div>

      <div class="group">
        <div class="label">过程类型</div>
        <select id="mode">
          <option value="generator">迷宫生成过程</option>
          <option value="resource">最佳路径生成过程</option>
        </select>
      </div>

      <div class="group" id="algo-group">
        <div class="label">算法选择</div>
        <select id="algorithm"></select>
      </div>

      <div class="group">
        <div class="label">播放控制</div>
        <div class="controls">
          <button id="prev">上一步</button>
          <button id="play">播放</button>
          <button id="next">下一步</button>
        </div>
      </div>

      <div class="group">
        <div class="label">速度</div>
        <input id="speed" type="range" min="80" max="1200" step="40" value="420" />
      </div>

      <div class="group meta">
        <h2 id="frame-title">加载中</h2>
        <p id="frame-desc">正在读取可视化数据。</p>
      </div>
    </aside>

    <main class="panel stage">
      <div class="stage-head">
        <h2 id="stage-title">过程帧</h2>
        <span id="stage-meta">第 0 / 0 步</span>
      </div>
      <div class="canvas-wrap">
        <canvas id="maze-canvas" width="960" height="960"></canvas>
      </div>
      <div class="timeline">
        <span class="tiny" id="start-step">1</span>
        <input id="slider" type="range" min="1" max="1" value="1" />
        <span class="tiny" id="end-step">1</span>
      </div>
    </main>
  </div>

  <script>
    const LIVE_MODE = {live_flag};
    const canvas = document.getElementById("maze-canvas");
    const ctx = canvas.getContext("2d");
    const modeEl = document.getElementById("mode");
    const algoEl = document.getElementById("algorithm");
    const sliderEl = document.getElementById("slider");
    const playEl = document.getElementById("play");
    const prevEl = document.getElementById("prev");
    const nextEl = document.getElementById("next");
    const speedEl = document.getElementById("speed");
    const frameTitleEl = document.getElementById("frame-title");
    const frameDescEl = document.getElementById("frame-desc");
    const stageTitleEl = document.getElementById("stage-title");
    const stageMetaEl = document.getElementById("stage-meta");
    const startStepEl = document.getElementById("start-step");
    const endStepEl = document.getElementById("end-step");
    const livePanelEl = document.getElementById("live-panel");

    const palette = {{
      "#": css("--wall"),
      " ": css("--road"),
      ".": css("--road"),
      "S": css("--start"),
      "E": css("--exit"),
      "B": css("--boss"),
      "G": css("--gold"),
      "T": css("--trap")
    }};

    let bundle = null;
    let playing = false;
    let timer = null;
    let currentMode = "generator";
    let currentAlgorithm = null;
    let currentIndex = 0;

    livePanelEl.style.display = LIVE_MODE ? "flex" : "none";

    function css(name) {{
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }}

    function normalizeSize(n) {{
      let value = Number.isFinite(n) ? Math.floor(n) : 15;
      value = Math.max(5, value);
      if (value % 2 === 0) value += 1;
      return value;
    }}

    function frameSet() {{
      if (!bundle) return [];
      if (currentMode === "resource") return bundle.resource_process?.frames || [];
      const found = (bundle.algorithms || []).find(item => item.name === currentAlgorithm);
      return found ? found.frames : [];
    }}

    function populateAlgorithms() {{
      algoEl.innerHTML = "";
      (bundle.algorithms || []).forEach(item => {{
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = item.name;
        algoEl.appendChild(option);
      }});
      currentAlgorithm = (bundle.algorithms || [])[0]?.name || null;
      if (currentAlgorithm) algoEl.value = currentAlgorithm;
    }}

    function updateControls() {{
      const frames = frameSet();
      const max = Math.max(1, frames.length);
      sliderEl.min = 1;
      sliderEl.max = max;
      sliderEl.value = Math.min(currentIndex + 1, max);
      startStepEl.textContent = "1";
      endStepEl.textContent = String(max);
      document.getElementById("algo-group").style.display = currentMode === "generator" ? "flex" : "none";
    }}

    function render() {{
      const frames = frameSet();
      if (!frames.length) {{
        frameTitleEl.textContent = "暂无帧";
        frameDescEl.textContent = "当前过程没有可展示的帧。";
        return;
      }}
      currentIndex = Math.max(0, Math.min(currentIndex, frames.length - 1));
      const frame = frames[currentIndex];
      drawFrame(frame);
      sliderEl.value = currentIndex + 1;
      frameTitleEl.textContent = frame.title || "Step";
      frameDescEl.textContent = frame.description || "";
      stageTitleEl.textContent = currentMode === "generator" ? `迷宫生成: ${{currentAlgorithm}}` : "最佳路径生成过程";
      stageMetaEl.textContent = `第 ${{currentIndex + 1}} / ${{frames.length}} 步`;
    }}

    function drawFrame(frame) {{
      const grid = frame.grid;
      const rows = grid.length;
      const cols = grid[0].length;
      const cell = Math.min(canvas.width / cols, canvas.height / rows);
      const offsetX = (canvas.width - cols * cell) / 2;
      const offsetY = (canvas.height - rows * cell) / 2;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = css("--paper");
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const x = offsetX + c * cell;
          const y = offsetY + r * cell;
          const value = grid[r][c];
          ctx.fillStyle = palette[value] || css("--road");
          ctx.fillRect(x, y, cell, cell);
          ctx.strokeStyle = "rgba(23,32,51,0.08)";
          ctx.strokeRect(x, y, cell, cell);
          if (value !== "#" && value !== " " && value !== ".") {{
            ctx.fillStyle = value === "G" ? "#172033" : "#ffffff";
            ctx.font = `${{Math.max(11, cell * 0.44)}}px Georgia`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(value, x + cell / 2, y + cell / 2);
          }}
        }}
      }}

      if (frame.path && frame.path.length > 1) {{
        ctx.strokeStyle = css("--path");
        ctx.lineWidth = Math.max(2, cell * 0.14);
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        frame.path.forEach((point, index) => {{
          const [r, c] = point;
          const x = offsetX + c * cell + cell / 2;
          const y = offsetY + r * cell + cell / 2;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();
      }}

      (frame.highlights || []).forEach(point => {{
        const [r, c] = point;
        const x = offsetX + c * cell + cell / 2;
        const y = offsetY + r * cell + cell / 2;
        ctx.beginPath();
        ctx.fillStyle = "rgba(198,95,33,0.26)";
        ctx.arc(x, y, cell * 0.36, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = css("--accent");
        ctx.lineWidth = Math.max(2, cell * 0.07);
        ctx.stroke();
      }});
    }}

    function stop() {{
      playing = false;
      playEl.textContent = "播放";
      if (timer) clearInterval(timer);
      timer = null;
    }}

    function start() {{
      stop();
      playing = true;
      playEl.textContent = "暂停";
      timer = setInterval(() => {{
        const frames = frameSet();
        if (currentIndex >= frames.length - 1) {{
          stop();
          return;
        }}
        currentIndex += 1;
        render();
      }}, Number(speedEl.value));
    }}

    async function loadBundle(url) {{
      frameTitleEl.textContent = "生成中";
      frameDescEl.textContent = "正在计算迷宫和最佳路径过程...";
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
      bundle = await resp.json();
      populateAlgorithms();
      currentMode = "generator";
      modeEl.value = "generator";
      currentIndex = 0;
      updateControls();
      render();
    }}

    modeEl.addEventListener("change", () => {{
      currentMode = modeEl.value;
      currentIndex = 0;
      stop();
      updateControls();
      render();
    }});

    algoEl.addEventListener("change", () => {{
      currentAlgorithm = algoEl.value;
      currentIndex = 0;
      stop();
      updateControls();
      render();
    }});

    sliderEl.addEventListener("input", () => {{
      currentIndex = Number(sliderEl.value) - 1;
      render();
    }});

    playEl.addEventListener("click", () => playing ? stop() : start());
    prevEl.addEventListener("click", () => {{
      stop();
      currentIndex = Math.max(0, currentIndex - 1);
      render();
    }});
    nextEl.addEventListener("click", () => {{
      stop();
      currentIndex = Math.min(frameSet().length - 1, currentIndex + 1);
      render();
    }});
    speedEl.addEventListener("input", () => {{
      if (playing) start();
    }});

    if (LIVE_MODE) {{
      document.getElementById("regenerate").addEventListener("click", async () => {{
        stop();
        const n = normalizeSize(Number(document.getElementById("size-input").value));
        const seed = Math.max(0, Math.floor(Number(document.getElementById("seed-input").value) || 0));
        document.getElementById("size-input").value = n;
        await loadBundle(`/api/bundle?n=${{n}}&seed=${{seed}}`);
      }});
      document.getElementById("single-regenerate").addEventListener("click", async () => {{
        stop();
        const n = normalizeSize(Number(document.getElementById("size-input").value));
        const seed = Math.max(0, Math.floor(Number(document.getElementById("seed-input").value) || 0));
        const algorithm = algoEl.value || "bfs_expansion";
        document.getElementById("size-input").value = n;
        await loadBundle(`/api/single?n=${{n}}&seed=${{seed}}&algorithm=${{encodeURIComponent(algorithm)}}`);
      }});
      loadBundle("/api/bundle?n=15&seed=42").catch(err => {{
        frameTitleEl.textContent = "加载失败";
        frameDescEl.textContent = String(err);
      }});
    }} else {{
      loadBundle("viewer_data.json").catch(err => {{
        frameTitleEl.textContent = "加载失败";
        frameDescEl.textContent = "请确认 viewer_data.json 与 viewer.html 在同一目录。 " + String(err);
      }});
    }}
  </script>
</body>
</html>
"""


