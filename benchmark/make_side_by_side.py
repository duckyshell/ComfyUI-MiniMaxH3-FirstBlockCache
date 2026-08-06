from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LEFT = RESULTS / "native_02_baseline_warm.mp4"
RIGHT = RESULTS / "native_04_fast_warm.mp4"
OUTPUT = RESULTS / "native_baseline_vs_fast_side_by_side.mp4"


def font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def duration(ffprobe: str, video: Path) -> float:
    raw = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        text=True,
    )
    return float(json.loads(raw)["format"]["duration"])


def main():
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe are required")

    with tempfile.TemporaryDirectory(prefix="h3_fbcache_sbs_") as temp_dir:
        header_path = Path(temp_dir) / "header.png"
        header = Image.new("RGB", (1920, 92), "#111318")
        draw = ImageDraw.Draw(header)
        title = "MiniMax H3 — native attention · fixed seed · 0.5 MP · 5 s · 20 steps"
        title_box = draw.textbbox((0, 0), title, font=font(25, True))
        draw.text(((1920 - (title_box[2] - title_box[0])) / 2, 9), title, fill="#f3f5f7", font=font(25, True))
        draw.text((24, 51), "NO CACHE — 90.64 s", fill="#56b4ff", font=font(22, True))
        draw.text((984, 51), "FAST — 60.82 s · 1.49× · −32.9%", fill="#85d996", font=font(22, True))
        header.save(header_path)

        clip_duration = min(duration(ffprobe, LEFT), duration(ffprobe, RIGHT))
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(LEFT), "-i", str(RIGHT),
                "-loop", "1", "-framerate", "24", "-i", str(header_path),
                "-filter_complex",
                "[0:v]setpts=PTS-STARTPTS[left];[1:v]setpts=PTS-STARTPTS[right];"
                "[left][right]hstack=inputs=2[stack];[2:v]format=yuv420p[header];"
                "[header][stack]vstack=inputs=2[out]",
                "-map", "[out]", "-map", "0:a?", "-t", f"{clip_duration:.6f}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-y", str(OUTPUT),
            ],
            check=True,
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
