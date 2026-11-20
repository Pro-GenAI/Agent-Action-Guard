"""Check both live package download badges on all public descriptions."""

from html import unescape
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
PYPI = "https://static.pepy.tech/personalized-badge/agent-action-guard?period=total&units=NONE&left_color=BLACK&right_color=GREEN&left_text=PyPI%20Downloads"
NPM = "https://img.shields.io/npm/d18m/agent-action-guard?style=for-the-badge&label=npm%20downloads"


class DownloadBadgeTests(unittest.TestCase):
    def test_package_readmes_include_pypi_and_npm(self):
        for path in ("python/README.md", "typescript/README.md"):
            with self.subTest(path=path):
                source = unescape((ROOT / path).read_text())
                self.assertIn(PYPI, source)
                self.assertIn(NPM, source)
                self.assertIn("https://pepy.tech/projects/agent-action-guard", source)
                self.assertIn("https://www.npmjs.com/package/agent-action-guard", source)

    def test_homepage_and_markdown_edition_include_both(self):
        homepage = unescape((ROOT / "docs/website/index.html").read_text())
        markdown = (ROOT / "docs/website/index.md").read_text()
        for url in (PYPI, NPM):
            self.assertIn(f'src="{url}"', homepage)
            self.assertIn(url, markdown)
        self.assertIn('alt="PyPI total downloads"', homepage)
        self.assertIn('alt="npm downloads in the last 18 months"', homepage)
        self.assertIn('class="download-badges enter"', homepage)
        self.assertIn('referrerpolicy="no-referrer"', homepage)

    def test_repo_readme_matches_badge_urls(self):
        source = (ROOT / "README.md").read_text()
        self.assertIn(PYPI, source)
        self.assertIn(NPM, source)


if __name__ == "__main__":
    unittest.main()
