"""Check public outbound site links; report blocked/time-out URLs as unverified."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
SITE_HOSTS = {"action-guard.github.io", "agent-leaderboard.github.io"}


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: set[str] = set()

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href", "")
        if tag != "a" or not href:
            return
        parsed = urlsplit(href)
        if parsed.scheme == "https" and parsed.hostname not in SITE_HOSTS:
            self.urls.add(urlunsplit(parsed._replace(fragment="")))


def check_url(url: str, timeout: float = 10) -> dict:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return {
            "url": url,
            "status": None,
            "state": "unverified",
            "detail": "Expected an HTTPS URL without credentials",
        }
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Agent-Action-Guard launch link check/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            gated = any(
                segment in urlsplit(final_url).path.split("/")
                for segment in ("challenge", "captcha", "authwall", "login")
            )
            return {
                "url": url,
                "status": response.status,
                "state": "unverified" if gated else "ok",
                "final_url": final_url,
            }
    except urllib.error.HTTPError as exc:
        return {
            "url": url,
            "status": exc.code,
            "state": "broken" if exc.code in {404, 410} else "unverified",
        }
    except (OSError, ValueError) as exc:
        return {"url": url, "status": None, "state": "unverified", "detail": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument(
        "--report", type=Path, default=ROOT / ".cache/site-launch/external-links.json"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    links = Links()
    for folder in ("docs/website", "docs/agent-leaderboard-site"):
        for page in (ROOT / folder).rglob("index.html"):
            links.feed(page.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(lambda url: check_url(url, args.timeout), sorted(links.urls))
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    counts = Counter(result["state"] for result in results)
    print(
        f"External links: {counts['ok']} OK, {counts['broken']} broken, {counts['unverified']} unverified"
    )
    for result in results:
        if result["state"] == "broken":
            print(f"  HTTP {result['status']}: {result['url']}")
    print(f"Full report: {args.report}")
    raise SystemExit(1 if counts["broken"] else 0)


if __name__ == "__main__":
    main()
