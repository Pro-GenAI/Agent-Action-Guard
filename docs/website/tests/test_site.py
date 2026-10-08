from __future__ import annotations

import json
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

SITE_ROOT = Path(__file__).resolve().parents[1]
HOME = SITE_ROOT / "index.html"
DOCS = SITE_ROOT / "docs" / "index.html"
LEADERBOARD = SITE_ROOT.parent / "agent-leaderboard-site" / "index.html"


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


class _AssetRefs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.local_assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {key: value or "" for key, value in attrs}
        ref = data.get("src") if tag == "script" else data.get("href") if tag == "link" else ""
        if ref and not ref.startswith(("http://", "https://", "//", "/", "#")):
            if tag == "script" or data.get("rel") == "stylesheet":
                self.local_assets.append(ref)


def parse(path: Path) -> _HTMLFacts:
    parser = _HTMLFacts()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def page_with_assets(path: Path) -> str:
    """Return a page's HTML plus any local stylesheets/scripts it links to."""
    facts = _AssetRefs()
    html = path.read_text(encoding="utf-8")
    facts.feed(html)
    parts = [html]
    for ref in facts.local_assets:
        parts.append((path.parent / ref).read_text(encoding="utf-8"))
    return "\n".join(parts)


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
        home = page_with_assets(HOME)
        docs = page_with_assets(DOCS)
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

    def test_homepage_css_and_js_live_in_linked_assets(self) -> None:
        html = HOME.read_text(encoding="utf-8")
        refs = _AssetRefs()
        refs.feed(html)
        self.assertEqual(
            sorted(refs.local_assets), ["assets/home.css", "assets/home.js"]
        )
        for ref in refs.local_assets:
            with self.subTest(asset=ref):
                self.assertTrue((SITE_ROOT / ref).read_text(encoding="utf-8").strip())
        self.assertNotIn("<style", html)
        # Only tiny bootstrap/JSON-LD scripts may stay inline.
        inline = [
            body
            for body in re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
            if body.strip()
        ]
        self.assertTrue(all(len(body) < 2500 for body in inline))


    def test_supported_hook_targets_are_documented(self) -> None:
        home = HOME.read_text(encoding="utf-8")
        docs = DOCS.read_text(encoding="utf-8")
        llms = (SITE_ROOT / "llms.txt").read_text(encoding="utf-8")

        targets = {
            "codex": "Codex",
            "claude-code": "Claude Code",
            "cursor": "Cursor",
            "kiro": "Kiro",
            "opencode": "OpenCode",
            "agy": "Antigravity CLI",
            "copilot": "GitHub Copilot CLI",
            "openclaw": "OpenClaw",
            "hermes": "Hermes Agent",
        }

        for target, label in targets.items():
            with self.subTest(target=target):
                self.assertIn(label, home)
                self.assertIn(
                    f"agent-action-guard hooks install --target {target}",
                    docs,
                )
                self.assertIn(f"`{target}`", llms)

        self.assertIn(
            "openclaw plugins enable agent-action-guard",
            docs,
        )
        self.assertIn("HERMES_ENABLE_PROJECT_PLUGINS=1", docs)
        self.assertIn(
            ".openclaw/extensions/agent-action-guard/",
            docs,
        )
        self.assertIn(
            ".hermes/plugins/agent-action-guard/",
            docs,
        )

    def test_homepage_benchmark_numbers_match_readme(self) -> None:
        readme = (SITE_ROOT.parents[1] / "README.md").read_text(encoding="utf-8")
        home = HOME.read_text(encoding="utf-8")
        for value in ("97.87%", "98.92%", "19.66", "3381.72", "815.86", "90.07%"):
            with self.subTest(value=value):
                self.assertIn(value, readme)
                self.assertIn(value, home)

    def test_homepage_benchmark_links_to_full_results(self) -> None:
        home = HOME.read_text(encoding="utf-8")
        facts = parse(HOME)

        self.assertIn("View full benchmark results", home)
        self.assertIn("https://agent-leaderboard.github.io/", facts.links)
        problem_section = re.search(
            r'<section id="problem".*?</section>',
            home,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(problem_section)
        self.assertIn(
            "https://agent-leaderboard.github.io/",
            problem_section.group(0),
        )

    def test_harmactionseval_snapshot_is_synchronized(self) -> None:
        readme = (SITE_ROOT.parents[1] / "README.md").read_text(encoding="utf-8")
        python_readme = (SITE_ROOT.parents[1] / "python" / "README.md").read_text(
            encoding="utf-8"
        )
        home = HOME.read_text(encoding="utf-8")
        leaderboard = LEADERBOARD.read_text(encoding="utf-8")

        scores = [
            float(value)
            for value in re.findall(
                r'<td[^>]*text-lg[^>]*>([0-9]+\.[0-9]+)%</td>',
                leaderboard,
            )
        ]
        ranks = re.findall(
            r'<td class="px-8 py-5 text-sm font-headline font-bold '
            r'text-on-surface/60">(\d{2})</td>',
            leaderboard,
        )
        self.assertEqual(ranks, [f"{rank:02d}" for rank in range(1, 20)])
        self.assertEqual(len(scores), 19)

        average = sum(scores) / len(scores)
        top_score = max(scores)
        high_execution_share = round(
            100 * sum(score < 5.0 for score in scores) / len(scores)
        )

        self.assertAlmostEqual(average, 11.91, places=2)
        self.assertEqual(top_score, 79.43)
        self.assertEqual(high_execution_share, 68)

        for value in ("Gemini 3.8 Flash", f"{top_score:.2f}%", f"{average:.2f}%"):
            with self.subTest(value=value):
                self.assertIn(value, readme)
                self.assertIn(value, python_readme)
                self.assertIn(value, home)
                self.assertIn(value, leaderboard)

        execution_share = f"{high_execution_share}%"
        self.assertIn(execution_share, readme)
        self.assertIn(execution_share, python_readme)
        self.assertIn(execution_share, home)

        model_count = len(scores)
        average_label = f"All {model_count} model average"
        self.assertIn(average_label, readme)
        self.assertIn(average_label, python_readme)
        self.assertIn(average_label, home)
        self.assertLess(
            leaderboard.index("Gemini 3.8 Flash"),
            leaderboard.index("Claude Opus 5"),
        )
        self.assertIn(f"{model_count} models", leaderboard)
        self.assertIn(
            f"<strong>{model_count}</strong><span>Models tested</span>",
            leaderboard,
        )
        self.assertIn(f'(value / {top_score:.2f}) * 100', leaderboard)


if __name__ == "__main__":
    unittest.main()
