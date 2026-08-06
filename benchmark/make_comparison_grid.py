from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MATRIX = RESULTS / "matrix_results_sage2.json"
OUTPUT = RESULTS / "sage2_warm_comparison_grid.jpg"
TIMESTAMPS = (0.50, 1.75, 3.00, 4.25)
ROWS = (
    ("baseline", "sage2_02_baseline_warm.mp4"),
    ("safe", "sage2_04_safe_warm.mp4"),
    ("fast", "sage2_06_fast_warm.mp4"),
    ("aggressive", "sage2_08_aggressive_warm.mp4"),
)


def font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def extract_frame(ffmpeg: str, video: Path, timestamp: float, destination: Path):
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(timestamp), "-i", str(video),
         "-frames:v", "1", "-vf", "scale=400:-2", "-y", str(destination)],
        check=True,
    )


def main():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    runs = json.loads(MATRIX.read_text(encoding="utf-8"))
    timings = {row["config"]: row["wall_seconds"] for row in runs if row["temperature"] == "warm"}
    baseline = timings["baseline"]

    cell_w, cell_h = 400, 226
    label_w, top_h, row_gap = 240, 82, 8
    canvas = Image.new("RGB", (label_w + cell_w * len(TIMESTAMPS), top_h + (cell_h + row_gap) * len(ROWS)), "#111318")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "MiniMax H3 FirstBlockCache — SageAttention2 warm A/B", fill="#f3f5f7", font=font(25, True))
    draw.text((18, 46), "0.5 MP · 5 s · 20 steps · fixed seed 20260807 · identical backend", fill="#9ba7b4", font=font(17))
    for column, timestamp in enumerate(TIMESTAMPS):
        draw.text((label_w + column * cell_w + 10, 56), f"{timestamp:.2f} s", fill="#c9d2dc", font=font(16, True))

    with tempfile.TemporaryDirectory(prefix="h3_fbcache_grid_") as temp_dir:
        temp = Path(temp_dir)
        for row_index, (config, filename) in enumerate(ROWS):
            y = top_h + row_index * (cell_h + row_gap)
            seconds = timings[config]
            speedup = baseline / seconds
            reduction = (1.0 - seconds / baseline) * 100.0
            title = "No cache" if config == "baseline" else config.capitalize()
            draw.text((18, y + 60), title, fill="#f3f5f7", font=font(22, True))
            draw.text((18, y + 93), f"{seconds:.2f} s", fill="#56b4ff", font=font(20, True))
            if config != "baseline":
                draw.text((18, y + 123), f"{speedup:.2f}× · −{reduction:.1f}%", fill="#85d996", font=font(17))
            for column, timestamp in enumerate(TIMESTAMPS):
                frame_path = temp / f"{row_index}_{column}.png"
                extract_frame(ffmpeg, RESULTS / filename, timestamp, frame_path)
                with Image.open(frame_path) as frame:
                    frame = frame.convert("RGB")
                    if frame.size != (cell_w, cell_h):
                        frame = frame.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                    canvas.paste(frame, (label_w + column * cell_w, y))

    canvas.save(OUTPUT, quality=94, subsampling=0)
    print(OUTPUT)


if __name__ == "__main__":
    main()
