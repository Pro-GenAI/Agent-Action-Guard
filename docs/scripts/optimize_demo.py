"""Create a lossless WebP demo and prove every decoded frame/timing is preserved."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageSequence

assets = Path(__file__).resolve().parents[1] / "website/assets"
source_path, output_path = assets / "demo.gif", assets / "demo.webp"
with Image.open(source_path) as source:
    frames = [frame.convert("RGBA").copy() for frame in ImageSequence.Iterator(source)]
    durations = []
    for index in range(source.n_frames):
        source.seek(index)
        durations.append(source.info.get("duration", 100))
    loop = source.info.get("loop", 0)
buffer = BytesIO()
frames[0].save(
    buffer,
    format="WEBP",
    lossless=True,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=loop,
    method=6,
)
payload = buffer.getvalue()
digest = hashlib.sha256()
with Image.open(BytesIO(payload)) as optimized:
    if optimized.n_frames != len(frames):
        raise ValueError("Optimization changed the animation frame count")
    for index, frame in enumerate(frames):
        optimized.seek(index)
        optimized.load()
        if frame.tobytes() != optimized.convert("RGBA").tobytes() or durations[
            index
        ] != optimized.info.get("duration"):
            raise ValueError(f"Optimization changed pixels or timing in frame {index}")
        digest.update(frame.tobytes())
if len(payload) >= source_path.stat().st_size:
    raise ValueError("Optimized image is not smaller; retain the original format")
manifest = {
    "source": source_path.name,
    "optimized": output_path.name,
    "width": frames[0].width,
    "height": frames[0].height,
    "frames": len(frames),
    "durations_ms": durations,
    "loop": loop,
    "rgba_frames_sha256": digest.hexdigest(),
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "optimized_sha256": hashlib.sha256(payload).hexdigest(),
}
output_path.write_bytes(payload)
(assets / "demo-optimization.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    f"Verified {len(frames)} frames; saved {source_path.stat().st_size - output_path.stat().st_size} bytes"
)
