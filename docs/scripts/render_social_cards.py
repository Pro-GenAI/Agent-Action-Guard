"""Render the editable SVG social cards. Dev-only dependencies: cairosvg, pillow."""

from io import BytesIO
from pathlib import Path

import cairosvg
from PIL import Image

DOCS = Path(__file__).resolve().parents[1]
for site in ("website", "agent-leaderboard-site"):
    source = DOCS / site / "assets/social-card.svg"
    png = cairosvg.svg2png(url=str(source))
    with Image.open(BytesIO(png)) as image:
        image.convert("RGB").save(
            source.with_suffix(".jpg"), quality=90, optimize=True, progressive=True
        )
    print(source.with_suffix(".jpg"))
