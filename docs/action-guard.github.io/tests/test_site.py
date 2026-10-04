from __future__ import annotations

import json
import unittest
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

SITE_ROOT = Path(__file__).resolve().parents[1]
HOME = SITE_ROOT / "index.html"
DOCS = SITE_ROOT / "docs" / "index.html"


class _HTMLFacts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.metas: list[dict[str, str]] = []
        self.canonicals: list[str] = []
        self.titles: list[str] = []
        self.h1_count = 0
        self._in_title = False
        self._json_ld = False
        self.json_ld_blocks: list[str] = []
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {key: value or "" for key, value in attrs}
        if tag == "a" and data.get("href"):
            self.links.append(data["href"])
        elif tag == "meta":
            self.metas.append(data)
        elif tag == "link" and data.get("rel") == "canonical":
            self.canonicals.append(data.get("href", ""))
        elif tag == "title":
            self._in_title = True
        elif tag == "h1":
            self.h1_count += 1
        elif tag == "script" and data.get("type") == "application/ld+json":
            self._json_ld = True
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._json_ld:
            self._json_ld = False
            self.json_ld_blocks.append("".join(self._buffer))

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.titles.append(data)
        if self._json_ld:
            self._buffer.append(data)


def parse(path: Path) -> _HTMLFacts:
    parser = _HTMLFacts()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def meta_value(facts: _HTMLFacts, key: str, value: str) -> str | None:
    for item in facts.metas:
        if item.get(key) == value:
            return item.get("content")
    return None


class SiteSeoTests(unittest.TestCase):
    def test_public_pages_have_core_seo_metadata(self) -> None:
        cases = [
            (HOME, "https://action-guard.github.io/"),
            (DOCS, "https://action-guard.github.io/docs/"),
        ]
        for path, canonical in cases:
            with self.subTest(path=path):
                facts = parse(path)
                self.assertTrue("".join(facts.titles).strip())
                self.assertEqual(facts.h1_count, 1)
                self.assertIn(canonical, facts.canonicals)
                self.assertTrue(meta_value(facts, "name", "description"))
                self.assertTrue(meta_value(facts, "property", "og:title"))
                self.assertTrue(meta_value(facts, "property", "og:description"))
                self.assertTrue(meta_value(facts, "name", "twitter:card"))
                self.assertTrue(facts.json_ld_blocks)
                for block in facts.json_ld_blocks:
                    json.loads(block)

    def test_homepage_and_docs_are_mutually_linked(self) -> None:
        home = parse(HOME)
        docs = parse(DOCS)
        self.assertIn("/docs/", home.links)
        self.assertIn("/", docs.links)

    def test_readme_links_to_docs(self) -> None:
        readme = (SITE_ROOT.parents[1] / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://action-guard.github.io/docs/", readme)

    def test_robots_and_sitemap_expose_public_pages(self) -> None:
        robots = (SITE_ROOT / "robots.txt").read_text(encoding="utf-8")
        self.assertIn("User-agent: *", robots)
        self.assertIn("Allow: /", robots)
        self.assertIn("Sitemap: https://action-guard.github.io/sitemap.xml", robots)

        root = ElementTree.parse(SITE_ROOT / "sitemap.xml").getroot()
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = {node.text for node in root.findall("sm:url/sm:loc", namespace)}
        self.assertEqual(
            urls,
            {
                "https://action-guard.github.io/",
                "https://action-guard.github.io/docs/",
            },
        )

    def test_code_examples_use_language_aware_syntax_highlighting(self) -> None:
        home = HOME.read_text(encoding="utf-8")
        docs = DOCS.read_text(encoding="utf-8")
        highlight_src = (
            "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"
        )

        for name, page in (("home", home), ("docs", docs)):
            with self.subTest(page=name):
                self.assertIn(highlight_src, page)
                self.assertIn(".hljs-keyword", page)
                self.assertIn(".hljs-string", page)

        self.assertIn('<code class="language-python">', home)
        self.assertIn('<code class="language-javascript">', home)
        self.assertIn('<code class="language-python">', docs)
        self.assertIn('<code class="language-javascript">', docs)
        self.assertIn('<code class="language-bash">', docs)
        self.assertIn("highlightElement(code)", home)
        self.assertIn("highlightAll()", docs)


if __name__ == "__main__":
    unittest.main()
