"""Regression checks for the sites' same-origin, collection-free design."""

from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

DOCS_ROOT = Path(__file__).resolve().parents[2]
SITES = {
    "action-guard.github.io": DOCS_ROOT / "website",
    "agent-leaderboard.github.io": DOCS_ROOT / "agent-leaderboard-site",
}


class Resources(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []
        self.tags: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        self.tags.append(tag)
        if tag == "a":
            self.links.append(data.get("href", ""))
        if tag == "link" and set(data.get("rel", "").split()) & {
            "stylesheet",
            "preload",
            "modulepreload",
            "prefetch",
            "preconnect",
            "dns-prefetch",
            "icon",
            "manifest",
        }:
            self.assets.append(data.get("href", ""))
        if tag in {
            "script",
            "img",
            "iframe",
            "source",
            "video",
            "audio",
            "embed",
            "input",
        } and data.get("src"):
            self.assets.append(data["src"])
        if data.get("poster"):
            self.assets.append(data["poster"])
        if tag == "object" and data.get("data"):
            self.assets.append(data["data"])
        if data.get("srcset"):
            self.assets.extend(
                item.strip().split()[0] for item in data["srcset"].split(",")
            )
        self.assets.extend(css_urls(data.get("style", "")))


def css_urls(text):
    return [match[1] for match in re.findall(r"url\(\s*(['\"]?)(.*?)\1\s*\)", text)]


class PrivacyTests(unittest.TestCase):
    def test_privacy_notices_and_styles_stay_synchronized(self):
        main, leaderboard = SITES.values()
        self.assertEqual(
            (main / "privacy/index.html").read_text().split("<body>", 1)[1],
            (leaderboard / "privacy/index.html").read_text().split("<body>", 1)[1],
        )
        self.assertEqual(
            (main / "assets/privacy.css").read_bytes(),
            (leaderboard / "assets/privacy.css").read_bytes(),
        )

    def test_internal_page_links_resolve(self):
        for root in SITES.values():
            for page in root.rglob("index.html"):
                facts = Resources()
                facts.feed(page.read_text())
                for link in facts.links:
                    parsed = urlsplit(link)
                    if parsed.scheme or parsed.netloc or not parsed.path:
                        continue
                    path = (
                        root / unquote(parsed.path.lstrip("/"))
                        if link.startswith("/")
                        else page.parent / unquote(parsed.path)
                    )
                    with self.subTest(page=page, link=link):
                        self.assertTrue(path.exists())
                        if path.is_dir():
                            self.assertTrue((path / "index.html").is_file())

    def assert_local_asset(self, ref, page, root):
        if ref.startswith(("data:", "#")):
            return
        parsed = urlsplit(ref)
        self.assertFalse(
            parsed.scheme or parsed.netloc, f"Remote asset in {page}: {ref}"
        )
        path = (
            root / unquote(parsed.path.lstrip("/"))
            if ref.startswith("/")
            else page.parent / unquote(parsed.path)
        )
        self.assertTrue(path.resolve().is_relative_to(root.resolve()), ref)
        self.assertTrue(path.is_file(), f"Missing asset in {page}: {ref}")
        self.assertGreater(path.stat().st_size, 0)

    def test_all_browser_assets_are_local_and_exist(self):
        for root in SITES.values():
            for page in root.rglob("*.html"):
                with self.subTest(page=page):
                    html = page.read_text()
                    facts = Resources()
                    facts.feed(html)
                    for ref in facts.assets + css_urls(html):
                        self.assert_local_asset(ref, page, root)
            for css in root.rglob("*.css"):
                with self.subTest(css=css):
                    text = css.read_text()
                    self.assertNotRegex(text, r"@import\s+['\"]")
                    for ref in css_urls(text):
                        self.assert_local_asset(ref, css, root)

    def test_visible_pages_link_privacy_and_suppress_referrers(self):
        for root in SITES.values():
            for page in root.rglob("index.html"):
                with self.subTest(page=page):
                    html = page.read_text()
                    facts = Resources()
                    facts.feed(html)
                    if page.parent.name != "privacy":
                        self.assertIn("/privacy/", facts.links)
                    self.assertIn('<meta name="referrer" content="no-referrer"', html)
                    self.assertNotIn("form", facts.tags)
                    self.assertNotIn("iframe", facts.tags)

    def test_authored_code_has_no_tracking_storage_or_transmission(self):
        for root in SITES.values():
            files = list(root.rglob("*.html")) + list(root.rglob("*.js"))
            for path in files:
                if "vendor" in path.parts:
                    continue
                with self.subTest(path=path):
                    text = path.read_text()
                    self.assertNotRegex(
                        text, r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\("
                    )
                    self.assertNotRegex(
                        text,
                        r"\b(?:sendBeacon|localStorage|sessionStorage)\b|document\.cookie",
                    )
                    self.assertNotRegex(
                        text,
                        r"(?i)hotjar|fullstory|posthog|google-analytics|googletagmanager|rrweb",
                    )

    def test_fonts_and_libraries_keep_licenses(self):
        for root in SITES.values():
            fonts = list((root / "assets/fonts").glob("*.woff2"))
            self.assertTrue(fonts)
            for font in fonts:
                with self.subTest(font=font):
                    self.assertEqual(font.read_bytes()[:4], b"wOF2")
                    license_file = font.with_name(
                        font.name.replace("-latin.woff2", "-OFL.txt")
                    )
                    self.assertIn("SIL OPEN FONT LICENSE", license_file.read_text())
            self.assertIn(
                "font-display: swap", (root / "assets/fonts/fonts.css").read_text()
            )
            self.assertTrue((root / "assets/THIRD_PARTY.md").is_file())
        vendor = SITES["action-guard.github.io"] / "assets/vendor"
        self.assertIn("11.11.1", (vendor / "highlight-11.11.1.min.js").read_text())
        self.assertIn("Redistribution", (vendor / "highlight-LICENSE").read_text())
        assets = SITES["agent-leaderboard.github.io"] / "assets"
        self.assertIn(
            "Permission is hereby granted", (assets / "tailwind-LICENSE").read_text()
        )
        self.assertIn("CC0", (assets / "providers/simple-icons-LICENSE").read_text())
        self.assertNotIn(
            "tailwind.config =", (assets.parent / "index.html").read_text()
        )
        self.assertIn(".bg-background", (assets / "tailwind.css").read_text())

    def test_privacy_content_and_crawl_routes(self):
        for host, root in SITES.items():
            html = (root / "privacy/index.html").read_text()
            for detail in (
                "GitHub Pages",
                "AAG_EMBED_ONNX",
                "AAG_EMBED_GGUF",
                "EMBED_MODEL_NAME",
                "EMBEDDING_BASE_URL",
                "EMBEDDING_API_KEY",
                "OPENAI_API_KEY",
                "not your action text",
                "result caches",
                "mailto:praneeth.vad@gmail.com",
                "not represented as a registered DMCA agent",
            ):
                with self.subTest(host=host, detail=detail):
                    self.assertIn(detail, html)
            sitemap = ElementTree.parse(root / "sitemap.xml").getroot()
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            urls = {node.text for node in sitemap.findall("sm:url/sm:loc", ns)}
            self.assertIn(f"https://{host}/privacy/", urls)
            self.assertIn(
                f"Sitemap: https://{host}/sitemap.xml",
                (root / "robots.txt").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
