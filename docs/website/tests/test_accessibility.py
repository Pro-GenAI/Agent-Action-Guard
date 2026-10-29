"""Checks for the accessibility and policy contracts of the published sites."""

from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

DOCS_ROOT = Path(__file__).resolve().parents[2]
ROOTS = [DOCS_ROOT / "website", DOCS_ROOT / "agent-leaderboard-site"]


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def contrast(foreground, background):
    def rgb(value):
        return [int(value[index : index + 2], 16) / 255 for index in (1, 3, 5)]

    def luminance(value):
        linear = [
            v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            for v in rgb(value)
        ]
        return sum(v * w for v, w in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    a, b = sorted((luminance(foreground), luminance(background)))
    return (b + 0.05) / (a + 0.05)


class AccessibilityTests(unittest.TestCase):
    def test_markup_and_changed_assets_use_versioned_urls(self):
        for root in ROOTS:
            for page in root.rglob("index.html"):
                parser = Elements()
                parser.feed(page.read_text())
                for tag, attrs in parser.elements:
                    ref = (
                        attrs.get("src", "")
                        if tag == "script"
                        else attrs.get("href", "")
                        if tag == "link"
                        else ""
                    )
                    parsed = urlsplit(ref)
                    if Path(parsed.path).name in {
                        "home.js",
                        "home.css",
                        "privacy.css",
                        "tailwind.css",
                    }:
                        with self.subTest(page=page, asset=ref):
                            self.assertTrue(
                                parsed.query,
                                "Version changed assets to avoid mixed cached markup/scripts",
                            )

    def test_pages_have_skip_target_image_descriptions_and_policy_navigation(self):
        for root in ROOTS:
            for page in root.rglob("index.html"):
                with self.subTest(page=page):
                    parser = Elements()
                    parser.feed(page.read_text())
                    links = [attrs for tag, attrs in parser.elements if tag == "a"]
                    self.assertEqual(links[0].get("href"), "#main-content")
                    mains = [attrs for tag, attrs in parser.elements if tag == "main"]
                    self.assertEqual(len(mains), 1)
                    self.assertEqual(mains[0].get("id"), "main-content")
                    self.assertEqual(mains[0].get("tabindex"), "-1")
                    for href in [
                        "/terms/",
                        "/privacy/#cookies",
                        "/privacy/#data-requests",
                    ]:
                        self.assertIn(href, [link.get("href") for link in links])
                    for tag, attrs in parser.elements:
                        if tag == "img":
                            self.assertIn("alt", attrs)
                            if not attrs["alt"]:
                                self.assertEqual(attrs.get("aria-hidden"), "true")
                        if tag == "th":
                            self.assertIn(attrs.get("scope"), ["col", "row"])

    def test_tab_markup_has_accessible_panel_relationships_and_one_tab_stop(self):
        parser = Elements()
        parser.feed((ROOTS[0] / "index.html").read_text())
        by_id = {attrs["id"]: attrs for _, attrs in parser.elements if "id" in attrs}
        tablists = [
            attrs for _, attrs in parser.elements if attrs.get("role") == "tablist"
        ]
        tabs = [attrs for _, attrs in parser.elements if attrs.get("role") == "tab"]
        self.assertEqual(len(tablists), 3)
        self.assertEqual(len(tabs), 6)
        self.assertEqual(sum(tab.get("tabindex") == "0" for tab in tabs), 3)
        for tab in tabs:
            panel = by_id[tab["aria-controls"]]
            self.assertEqual(panel.get("role"), "tabpanel")
            self.assertEqual(panel.get("aria-labelledby"), tab["id"])
            self.assertEqual(panel.get("tabindex"), "0")
            active = tab.get("aria-selected") == "true"
            self.assertEqual(tab.get("tabindex"), "0" if active else "-1")
            self.assertEqual("hidden" in panel, not active)

    def test_muted_text_and_focus_colors_meet_minimum_contrast(self):
        css = (ROOTS[0] / "assets/home.css").read_text()
        tokens = dict(re.findall(r"--([\w-]+):\s*(#[\da-f]{6})", css))
        for foreground in ("dim", "muted", "text", "green-2"):
            for background in ("bg", "panel", "panel-2", "panel-3"):
                with self.subTest(foreground=foreground, background=background):
                    self.assertGreaterEqual(
                        contrast(tokens[foreground], tokens[background]), 4.5
                    )
        board = (ROOTS[1] / "index.html").read_text()
        for selector in ("stat-mini span", "live-pill", "podium-rank"):
            block = re.search(r"\." + re.escape(selector) + r"\s*\{([^}]+)\}", board)[1]
            color = re.search(r"color:\s*(#[\da-f]{6});", block)[1]
            self.assertGreaterEqual(contrast(color, "#292a2e"), 4.5)
        self.assertNotRegex(board, r"text-on-surface/(?:40|60)\b")
        for root in ROOTS:
            policy_css = (root / "assets/privacy.css").read_text()
            self.assertIn(".skip-link:focus", policy_css)
            self.assertIn(":focus-visible", policy_css)

    def test_claims_and_method_link_match_actual_project_scope(self):
        home = (ROOTS[0] / "index.html").read_text()
        for claim in (
            "Zero-Latency",
            "93%",
            "&lt;50ms",
            "intercepts tool outputs",
            "without sacrificing safety",
        ):
            self.assertNotIn(claim, home)
        self.assertIn("Classification can miss harmful actions", home)
        self.assertIn("not a guarantee for every workload", home)
        board = (ROOTS[1] / "index.html").read_text()
        self.assertNotIn("Full Report", board)
        self.assertIn('href="https://action-guard.github.io/docs/#evaluation"', board)
        self.assertIn('aria-label="View source on GitHub"', board)
        # Progressive enhancement leaves the benchmark visible when scripting is disabled.
        base_reveal = re.search(r"\n\s*\.reveal\s*\{([^}]+)\}", board)[1]
        self.assertIn("opacity: 1", base_reveal)

    def test_policy_details_and_terms_are_synchronized(self):
        for route in ("privacy", "terms"):
            main = (ROOTS[0] / route / "index.html").read_text()
            board = (ROOTS[1] / route / "index.html").read_text()
            self.assertEqual(main.split("<body>", 1)[1], board.split("<body>", 1)[1])
        for root in ROOTS:
            privacy = (root / "privacy/index.html").read_text()
            self.assertIn('id="cookies"', privacy)
            self.assertIn('id="data-requests"', privacy)
            self.assertIn("?subject=Agent%20Action%20Guard%20data%20request", privacy)
            self.assertIn("no user accounts or action-history database", privacy)
            terms = (root / "terms/index.html").read_text()
            self.assertIn('id="refunds"', terms)
            self.assertIn("no project purchase for us to refund", terms)
            self.assertIn("do not imply endorsement", terms)

    def test_copy_feedback_is_visible_and_announced_without_moving_focus(self):
        parser = Elements()
        parser.feed((ROOTS[0] / "index.html").read_text())
        by_id = {attrs["id"]: attrs for _, attrs in parser.elements if "id" in attrs}
        for name in ("install-copy-status", "code-copy-status"):
            attrs = by_id[name]
            self.assertEqual(attrs.get("role"), "status")
            self.assertEqual(attrs.get("aria-live"), "polite")
            self.assertEqual(attrs.get("aria-atomic"), "true")
            self.assertNotIn("hidden", attrs)
        css = (ROOTS[0] / "assets/home.css").read_text()
        self.assertIn(".copy-feedback { color: var(--muted)", css)

    def test_recognition_links_do_not_claim_unproven_stars_or_likes(self):
        html = (ROOTS[0] / "index.html").read_text()
        section = html.split('id="recognitions"', 1)[1].split("</section>", 1)[0]
        self.assertNotIn("GitHub Star", section)
        self.assertNotIn("stargazer", section)
        self.assertNotIn("LinkedIn like", section)
        self.assertNotIn("person-role", section)
        self.assertIn("Bibliography reference", section)
        self.assertIn("do not imply endorsement", section)
        self.assertIn("View post", section)
        self.assertIn("View feature", section)

    def test_policy_inventory_distinguishes_recipients_and_retention(self):
        for root in ROOTS:
            html = (root / "privacy/index.html").read_text()
            parser = Elements()
            parser.feed(html)
            ids = {attrs.get("id") for _, attrs in parser.elements}
            self.assertTrue({"data-collected", "third-parties", "runtime-data"} <= ids)
            for text in (
                "GitHub controls its hosting records and retention",
                "sender address, message, and any attachments",
                "There is no fixed deletion deadline promised here",
                "does not submit a message or add you to a marketing list",
                "Normal local classification does not send action payloads",
            ):
                self.assertIn(text, html)
            self.assertNotIn("form", [tag for tag, _ in parser.elements])
            terms = (root / "terms/index.html").read_text()
            self.assertIn("image credits or a bundled logo", terms)
            self.assertIn("Check the relevant asset's upstream license", terms)


if __name__ == "__main__":
    unittest.main()
