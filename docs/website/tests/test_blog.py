"""Static blog routes, editorial metadata, linking and asset contracts."""

from __future__ import annotations

import json
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / "blog"
SLUGS = (
    "unsafe-ai-agent-actions",
    "best-ai-agent-frameworks-2026",
    "stop-prompt-injection",
    "secure-mcp-tool-calls",
)
ORIGIN = "https://action-guard.pro-genai.com"


class PageFacts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.canonicals = []
        self.meta = []
        self.scripts = []
        self.stylesheets = []
        self.h1 = 0
        self.in_schema = False
        self.schema_text = ""
        self.schemas = []
        self.codes = []

    def handle_starttag(self, tag, attrs):
        props = dict(attrs)
        if props.get("id"):
            self.ids.add(props["id"])
        if tag == "a" and props.get("href"):
            self.links.append(props["href"])
        if tag == "h1":
            self.h1 += 1
        if tag == "link" and props.get("rel") == "canonical":
            self.canonicals.append(props.get("href"))
        if tag == "link" and props.get("rel") == "stylesheet":
            self.stylesheets.append(props.get("href"))
        if tag == "script" and props.get("src"):
            self.scripts.append(props.get("src"))
        if tag == "script" and props.get("type") == "application/ld+json":
            self.in_schema = True
            self.schema_text = ""
        if tag == "meta":
            self.meta.append(props)
        if tag == "code" and props.get("class"):
            self.codes.append(props["class"])

    def handle_endtag(self, tag):
        if tag == "script" and self.in_schema:
            self.in_schema = False
            self.schemas.append(json.loads(self.schema_text))

    def handle_data(self, text):
        if self.in_schema:
            self.schema_text += text


def parse(path):
    facts = PageFacts()
    facts.feed(path.read_text(encoding="utf-8"))
    return facts


def content_meta(facts, selector, name):
    return next((m.get("content") for m in facts.meta if m.get(selector) == name), None)


class BlogTests(unittest.TestCase):
    def test_index_links_to_each_real_article(self):
        facts = parse(BLOG / "index.html")
        self.assertEqual(facts.h1, 1)
        for slug in SLUGS:
            self.assertIn(f"/blog/{slug}/", facts.links)
            self.assertTrue((BLOG / slug / "index.html").is_file())

    def test_articles_use_valid_seo_and_article_structured_data(self):
        for slug in (None, *SLUGS):
            path = BLOG / slug / "index.html" if slug else BLOG / "index.html"
            route = f"/blog/{slug}/" if slug else "/blog/"
            with self.subTest(route=route):
                facts = parse(path)
                self.assertEqual(facts.h1, 1)
                self.assertEqual(facts.canonicals, [ORIGIN + route])
                self.assertEqual(content_meta(facts, "property", "og:url"), ORIGIN + route)
                self.assertEqual(content_meta(facts, "property", "og:image"), ORIGIN + "/assets/social-card.jpg")
                self.assertEqual(content_meta(facts, "name", "twitter:card"), "summary_large_image")
                self.assertTrue(content_meta(facts, "name", "description"))
                self.assertEqual(len(facts.schemas), 1)
                self.assertEqual(facts.schemas[0]["@type"], "BlogPosting" if slug else "Blog")
                self.assertEqual(facts.schemas[0]["url"], ORIGIN + route)
                if slug:
                    self.assertEqual(facts.schemas[0]["datePublished"], "2026-11-03")
                    self.assertEqual(facts.schemas[0]["mainEntityOfPage"], ORIGIN + route)

    def test_accessibility_and_internal_section_links(self):
        for slug in (None, *SLUGS):
            path = BLOG / slug / "index.html" if slug else BLOG / "index.html"
            html = path.read_text(encoding="utf-8")
            facts = parse(path)
            with self.subTest(path=path):
                self.assertLess(html.index('href="#main-content"'), html.index("<main"))
                self.assertIn("main-content", facts.ids)
                self.assertIn('tabindex="-1"', html)
                self.assertIn('aria-label="Footer"', html)
                for link in ("/", "/docs/", "/blog/", "/terms/", "/privacy/", "/privacy/#cookies", "/privacy/#data-requests"):
                    self.assertIn(link, facts.links)
                for href in facts.links:
                    if href.startswith("#"):
                        self.assertIn(href[1:], facts.ids)
                if slug:
                    self.assertIn('aria-label="On this page"', html)
                    self.assertIn('The problem', html)
                    self.assertIn('The solution', html)

    def test_assets_are_local_and_code_highlighting_uses_bundled_runtime(self):
        for slug in (None, *SLUGS):
            path = BLOG / slug / "index.html" if slug else BLOG / "index.html"
            facts = parse(path)
            for ref in facts.scripts + facts.stylesheets:
                with self.subTest(path=path, asset=ref):
                    self.assertFalse(urlsplit(ref).scheme or urlsplit(ref).netloc)
                    self.assertTrue((path.parent / urlsplit(ref).path).is_file())
            self.assertTrue(any("blog.css?v=" in ref for ref in facts.stylesheets))
            self.assertTrue(any("blog.js?v=" in ref for ref in facts.scripts))
            self.assertTrue(any("highlight-11.11.1.min.js" in ref for ref in facts.scripts))
            self.assertIn('name="referrer" content="no-referrer"', path.read_text())
            if slug in ("unsafe-ai-agent-actions", "secure-mcp-tool-calls"):
                self.assertIn("language-python", facts.codes)

    def test_article_categories_and_substantive_sections(self):
        for slug in SLUGS:
            html = (BLOG / slug / "index.html").read_text()
            self.assertGreaterEqual(html.count("<h2"), 5)
            self.assertIn('class="prose"', html)
            self.assertIn('href="/docs/', html)
        frameworks = (BLOG / "best-ai-agent-frameworks-2026/index.html").read_text()
        self.assertIn("Microsoft Agent Framework", frameworks)
        self.assertIn("archived", frameworks)
        self.assertIn('role="region" tabindex="0"', frameworks)


if __name__ == "__main__":
    unittest.main()
