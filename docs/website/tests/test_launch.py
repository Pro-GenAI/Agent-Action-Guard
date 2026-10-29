from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from test_site import meta_value, parse

REPO = Path(__file__).resolve().parents[3]
SITES = {
    "action-guard.github.io": REPO / "docs/website",
    "agent-leaderboard.github.io": REPO / "docs/agent-leaderboard-site",
}


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / f"docs/scripts/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backups = load_script("backup_sites")
external = load_script("check_external_links")


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a":
            self.links.append(attrs.get("href", ""))
        if tag == "img":
            self.images.append(attrs)


def facts(path):
    page = Page()
    page.feed(path.read_text())
    return page


class LaunchTests(unittest.TestCase):
    def test_public_privacy_inventory_matches_repository_source(self):
        for root in SITES.values():
            self.assertEqual(
                (root / "assets/privacy-review.md").read_bytes(),
                (REPO / "docs/PRIVACY_REVIEW.md").read_bytes(),
            )

    def test_site_links_and_fragments_resolve_across_both_origins(self):
        for root in SITES.values():
            for path in root.rglob("index.html"):
                for link in facts(path).links:
                    parsed = urlsplit(link)
                    if parsed.scheme or parsed.netloc:
                        if parsed.netloc not in SITES:
                            continue
                        target = SITES[parsed.netloc] / unquote(parsed.path.lstrip("/"))
                    else:
                        target = (
                            root / unquote(parsed.path.lstrip("/"))
                            if parsed.path.startswith("/")
                            else path.parent / unquote(parsed.path)
                            if parsed.path
                            else path
                        )
                    if target.is_dir():
                        target = target / "index.html"
                    with self.subTest(page=path, link=link):
                        self.assertTrue(target.is_file())
                        if parsed.fragment and target.suffix == ".html":
                            self.assertIn(unquote(parsed.fragment), facts(target).ids)

    def test_repository_markdown_links_have_targets(self):
        for name in (
            "README.md",
            "python/README.md",
            "typescript/README.md",
            "USAGE.md",
        ):
            path = REPO / name
            for ref in re.findall(r"\]\(([^)]*)\)", path.read_text()):
                with self.subTest(page=name, ref=ref):
                    self.assertTrue(
                        ref, "Do not create clickable badges with empty targets"
                    )
                    parsed = urlsplit(ref)
                    if parsed.scheme or parsed.netloc or not parsed.path:
                        continue
                    self.assertTrue((path.parent / unquote(parsed.path)).exists())

    def test_social_metadata_has_local_jpeg_cards_and_dimensions(self):
        for host, root in SITES.items():
            card = root / "assets/social-card.jpg"
            self.assertTrue(card.read_bytes().startswith(b"\xff\xd8"))
            self.assertLess(card.stat().st_size, 120_000)
            svg = (root / "assets/social-card.svg").read_text()
            self.assertIn('width="1200" height="630"', svg)
            for path in root.rglob("index.html"):
                page = parse(path)
                with self.subTest(page=path):
                    self.assertEqual(
                        meta_value(page, "property", "og:image"),
                        f"https://{host}/assets/social-card.jpg",
                    )
                    self.assertEqual(
                        meta_value(page, "name", "twitter:image"),
                        meta_value(page, "property", "og:image"),
                    )
                    self.assertEqual(
                        meta_value(page, "property", "og:image:type"), "image/jpeg"
                    )
                    self.assertEqual(
                        meta_value(page, "property", "og:image:width"), "1200"
                    )
                    self.assertEqual(
                        meta_value(page, "property", "og:image:height"), "630"
                    )
                    self.assertTrue(meta_value(page, "property", "og:image:alt"))
                    self.assertEqual(
                        meta_value(page, "name", "twitter:card"), "summary_large_image"
                    )
                    self.assertEqual(
                        meta_value(page, "property", "og:url"), page.canonicals[0]
                    )

    def test_assets_have_bounded_sizes_and_reserve_image_layout(self):
        for root in SITES.values():
            self.assertLess(
                sum(
                    p.stat().st_size
                    for p in (root / "assets").rglob("*")
                    if p.is_file()
                ),
                2 * 1024 * 1024,
            )
            for path in root.rglob("index.html"):
                for image in facts(path).images:
                    with self.subTest(page=path, image=image.get("src")):
                        self.assertGreater(int(image["width"]), 0)
                        self.assertGreater(int(image["height"]), 0)
                        self.assertEqual(image.get("loading"), "lazy")

    def test_optimized_demo_matches_the_verified_source_pair(self):
        assets = SITES["action-guard.github.io"] / "assets"
        manifest = json.loads((assets / "demo-optimization.json").read_text())
        for name, key in (
            ("source", "source_sha256"),
            ("optimized", "optimized_sha256"),
        ):
            self.assertEqual(
                hashlib.sha256((assets / manifest[name]).read_bytes()).hexdigest(),
                manifest[key],
            )
        self.assertLess(
            (assets / manifest["optimized"]).stat().st_size,
            (assets / manifest["source"]).stat().st_size,
        )
        self.assertEqual(len(manifest["durations_ms"]), manifest["frames"])
        self.assertEqual(manifest["frames"], 42)
        self.assertEqual(len(manifest["rgba_frames_sha256"]), 64)
        html = (assets.parent / "index.html").read_text()
        self.assertIn('srcset="assets/demo.webp" type="image/webp"', html)
        self.assertIn('src="assets/demo.gif"', html)

    def test_documented_runtime_matches_package_minimum(self):
        html = (SITES["action-guard.github.io"] / "docs/index.html").read_text()
        self.assertNotIn("Python 3.8", html)
        self.assertIn("Python 3.10 through 3.14", html)
        project = (REPO / "python/pyproject.toml").read_text()
        self.assertRegex(project, r'requires-python\s*=\s*"[^\"]*3\.10')

    def test_docs_contain_wide_code_and_tables_at_small_breakpoints(self):
        html = (SITES["action-guard.github.io"] / "docs/index.html").read_text()
        self.assertIn("grid-template-columns: minmax(0, 1fr)", html)
        self.assertIn("article, .card { min-width: 0; }", html)
        self.assertIn(".table-scroll { max-width: 100%; overflow-x: auto; }", html)
        self.assertEqual(html.count('class="table-scroll" role="region"'), 2)
        self.assertIn(
            'aria-label="Native coding-agent hook targets" tabindex="0"', html
        )


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / "docs/website"
        self.site.mkdir(parents=True)
        (self.site / "index.html").write_text("uncommitted source")
        self.output = self.root / ".cache/snapshot.zip"

    def test_snapshot_round_trip_includes_uncommitted_sources_and_skips_private_tools(
        self,
    ):
        for name in (
            "node_modules",
            "__pycache__",
            "playwright-report",
            "test-results",
        ):
            p = self.site / name
            p.mkdir()
            (p / "private.txt").write_text("excluded")
        (self.site / ".env.local").write_text("secret")
        count = backups.create_backup(self.root, self.output)
        self.assertEqual(count, 1)
        self.assertEqual(backups.verify_backup(self.output), count)
        with ZipFile(self.output) as archive:
            self.assertEqual(
                archive.read("docs/website/index.html"), b"uncommitted source"
            )

    def test_existing_snapshot_and_source_paths_are_preserved(self):
        backups.create_backup(self.root, self.output)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            backups.create_backup(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), before)
        with self.assertRaises(ValueError):
            backups.create_backup(self.root, self.site / "bad.zip")

    def test_concurrent_snapshot_creation_never_overwrites_the_winner(self):
        def create():
            try:
                return backups.create_backup(self.root, self.output)
            except FileExistsError:
                return "exists"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(lambda _: create(), range(2)))
        self.assertCountEqual(outcomes, [1, "exists"])
        self.assertEqual(backups.verify_backup(self.output), 1)

    def test_failed_verification_removes_only_the_new_partial_snapshot(self):
        with (
            patch.object(
                backups, "verify_backup", side_effect=ValueError("verification failed")
            ),
            self.assertRaises(ValueError),
        ):
            backups.create_backup(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual((self.site / "index.html").read_text(), "uncommitted source")

    def test_symlink_sources_are_rejected(self):
        outside = self.root / "private.txt"
        outside.write_text("private")
        try:
            (self.site / "link.txt").symlink_to(outside)
        except OSError:
            self.skipTest("Platform does not allow symlink creation")
        with self.assertRaises(ValueError):
            backups.create_backup(self.root, self.output)

    def test_corruption_and_extra_members_are_detected(self):
        backups.create_backup(self.root, self.output)
        with ZipFile(self.output, "a") as archive:
            archive.writestr("unexpected.txt", "not in manifest")
        with self.assertRaises(ValueError):
            backups.verify_backup(self.output)

    def test_checksum_mismatch_is_detected(self):
        self.output.parent.mkdir()
        with ZipFile(self.output, "w") as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "format": 1,
                        "files": [
                            {
                                "path": "docs/website/index.html",
                                "size": 3,
                                "sha256": "wrong",
                            }
                        ],
                    }
                ),
            )
            archive.writestr("docs/website/index.html", "new")
        with self.assertRaisesRegex(ValueError, "checksum"):
            backups.verify_backup(self.output)


class ExternalLinkTests(unittest.TestCase):
    def test_discovery_deduplicates_and_leaves_own_sites_for_local_checks(self):
        parser = external.Links()
        parser.feed(
            '<a href="https://example.com/a#one">one</a><a href="https://example.com/a#two">two</a><a href="https://action-guard.github.io/terms/">own</a><a href="mailto:a@example.com">email</a>'
        )
        self.assertEqual(parser.urls, {"https://example.com/a"})

    def test_success_and_blocked_broken_or_transient_responses(self):
        url = "https://example.com/"
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.geturl.return_value = url
        with patch.object(external.urllib.request, "urlopen", return_value=response):
            self.assertEqual(external.check_url(url)["state"], "ok")
        response.geturl.return_value = "https://example.com/challenge?redirect=profile"
        with patch.object(external.urllib.request, "urlopen", return_value=response):
            self.assertEqual(external.check_url(url)["state"], "unverified")
        for code in (404, 410, 403, 429, 503):
            with (
                self.subTest(code=code),
                patch.object(
                    external.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.HTTPError(url, code, "test", {}, None),
                ),
            ):
                self.assertEqual(
                    external.check_url(url)["state"],
                    "broken" if code in {404, 410} else "unverified",
                )
        with patch.object(
            external.urllib.request, "urlopen", side_effect=TimeoutError()
        ):
            self.assertEqual(external.check_url(url)["state"], "unverified")

    def test_credentials_and_non_https_urls_are_not_requested(self):
        with patch.object(external.urllib.request, "urlopen") as request:
            for url in (
                "file:///etc/passwd",
                "http://example.com/",
                "https://user:password@example.com/",
                "https:/missing-host",
            ):
                self.assertEqual(external.check_url(url)["state"], "unverified")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
