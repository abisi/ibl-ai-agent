"""Builds `report.html` (self-contained, base64-embedded images) from
`report_template.html` (source, plain relative `<img src="...">` paths into
`report_assets/`) -- keeps the git-tracked template small and diffable while
still producing a standalone file that opens directly in a browser with no
server. The published Claude-Artifact version of this report instead
references `report_assets/` files directly via the Artifact tool's
multi-file `files` map (no base64 needed there); this script is only for
the local standalone copy.

Usage: python build_report.py
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "report_template.html"
ASSETS_DIR = ROOT / "report_assets"
OUT_PATH = ROOT / "report.html"

MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}


def embed_images(html: str) -> str:
    def replace(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return match.group(0)
        path = ASSETS_DIR / src
        if not path.exists():
            raise FileNotFoundError(f"report_template.html references missing asset: {src}")
        mime = MIME_BY_SUFFIX.get(path.suffix.lower())
        if mime is None:
            raise ValueError(f"unknown image type for {src}")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', replace, html)


def main():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = embed_images(html)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH.name} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
