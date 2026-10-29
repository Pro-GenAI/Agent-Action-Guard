"""Generate/check content security policies for the two static GitHub Pages sites."""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITES = (ROOT / "docs/website", ROOT / "docs/agent-leaderboard-site")
CSP_META = re.compile(r'\s*<meta\s+http-equiv="Content-Security-Policy"[^>]*>', re.I)


class EventHandlers(HTMLParser):
    def __init__(self):
        super().__init__()
        self.handlers = set()

    def handle_starttag(self, tag, attrs):
        del tag
        self.handlers.update(
            value for name, value in attrs if name.startswith("on") and value
        )


def digest(source: str) -> str:
    return (
        "'sha256-"
        + base64.b64encode(hashlib.sha256(source.encode()).digest()).decode()
        + "'"
    )


def policy(source: str) -> str:
    scripts = {
        digest(body)
        for attrs, body in re.findall(
            r"<script\b([^>]*)>([\s\S]*?)</script\s*>", source, re.I
        )
        if not re.search(r"\bsrc\s*=", attrs, re.I) and body.strip()
    }
    handlers = EventHandlers()
    handlers.feed(source)
    scripts.update(digest(value) for value in handlers.handlers)
    if handlers.handlers:
        scripts.add("'unsafe-hashes'")
    return "; ".join(
        [
            "default-src 'none'",
            "base-uri 'none'",
            "object-src 'none'",
            "form-action 'none'",
            "frame-src 'none'",
            "connect-src 'none'",
            "font-src 'self'",
            "img-src 'self' data:",
            "media-src 'self'",
            "style-src 'self' 'unsafe-inline'",
            "script-src 'self' " + " ".join(sorted(scripts)),
        ]
    )


def render(source: str) -> str:
    source = CSP_META.sub("", source)
    meta = (
        '<meta http-equiv="Content-Security-Policy" content="' + policy(source) + '" />'
    )
    return re.sub(r'(<meta charset="utf-8"\s*/?>)', r"\1\n  " + meta, source, count=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail if a policy needs regeneration"
    )
    args = parser.parse_args()
    stale = []
    for root in SITES:
        for page in sorted(root.rglob("index.html")):
            source = page.read_text()
            expected = render(source)
            if source != expected:
                if args.check:
                    stale.append(str(page.relative_to(ROOT)))
                else:
                    page.write_text(expected)
    if stale:
        print(
            "Regenerate site CSP hashes: python3 docs/scripts/site_security.py\n"
            + "\n".join(stale)
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
