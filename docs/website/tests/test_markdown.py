"""Markdown mirrors and LLM discovery for both static sites."""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SITES = (
    (REPO / "docs" / "website", "https://action-guard.pro-genai.com/"),
    (REPO / "docs" / "agent-leaderboard-site", "https://agent-leaderboard.github.io/"),
)


class MarkdownDiscoverabilityTests(unittest.TestCase):
    def test_all_public_html_pages_have_discoverable_markdown_mirrors(self):
        root_index = (REPO / "llms.txt").read_text(encoding="utf-8")
        for site, origin in SITES:
            site_index = (site / "llms.txt").read_text(encoding="utf-8")
            self.assertTrue(site_index.startswith("# "))
            for page in sorted(site.rglob("index.html")):
                with self.subTest(page=str(page.relative_to(REPO))):
                    html = page.read_text(encoding="utf-8")
                    markdown_file = page.with_suffix(".md")
                    self.assertTrue(markdown_file.is_file())
                    markdown = markdown_file.read_text(encoding="utf-8")
                    relative = page.parent.relative_to(site).as_posix()
                    route = "/" if relative == "." else f"/{relative}/"
                    canonical = origin.rstrip("/") + route
                    md_url = canonical + "index.md"
                    self.assertIn(f'href="{canonical}"', html)
                    self.assertRegex(html, r'<link\s+rel="alternate"\s+type="text/markdown"')
                    self.assertIn(f'href="{md_url}"', html)
                    self.assertTrue(markdown.startswith("# "))
                    self.assertIn(f"]({canonical})", markdown)
                    self.assertGreater(len(markdown), 350)
                    for index in (site_index, root_index):
                        self.assertIn(f"]({canonical})", index)
                        self.assertIn(f"]({md_url})", index)

    def test_markdown_content_retains_docs_and_rankings(self):
        docs = (SITES[0][0] / "docs" / "index.md").read_text(encoding="utf-8")
        leaderboard = (SITES[1][0] / "index.md").read_text(encoding="utf-8")
        self.assertIn("```python", docs)
        self.assertIn("```javascript", docs)
        self.assertIn("SafeActions@1", leaderboard)
        self.assertIn("| Rank | Model Name |", leaderboard)


if __name__ == "__main__":
    unittest.main()
