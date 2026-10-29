"""Static deployment policies must match the exact scripts served to visitors."""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "site_security", ROOT / "docs/scripts/site_security.py"
)
security = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(security)


class SecurityTests(unittest.TestCase):
    def test_all_public_pages_have_current_restrictive_policies(self):
        for site in security.SITES:
            for page in site.rglob("index.html"):
                with self.subTest(page=page):
                    source = page.read_text()
                    self.assertEqual(source, security.render(source))
                    self.assertLess(
                        source.index("Content-Security-Policy"),
                        (
                            source.index("<script")
                            if "<script" in source
                            else source.index("</head>")
                        ),
                    )
                    policy = security.policy(source)
                    for control in [
                        "default-src 'none'",
                        "connect-src 'none'",
                        "base-uri 'none'",
                        "object-src 'none'",
                        "form-action 'none'",
                    ]:
                        self.assertIn(control, policy)
                    script_policy = policy.split("script-src ", 1)[1]
                    self.assertNotIn("'unsafe-inline'", script_policy)
                    self.assertNotIn("'unsafe-eval'", script_policy)

    def test_policy_hashes_detect_script_and_handler_changes(self):
        source = '<meta charset="utf-8" /><script>window.test=1;</script><img onerror="this.hidden=true">'
        rendered = security.render(source)
        self.assertEqual(security.render(rendered), rendered)
        self.assertIn(security.digest("window.test=1;"), rendered)
        self.assertIn(security.digest("this.hidden=true"), rendered)
        self.assertNotEqual(
            rendered, security.render(rendered.replace("test=1", "test=2"))
        )

    def test_external_scripts_are_allowed_only_from_self(self):
        source = '<script src="/assets/home.js"></script>'
        self.assertIn("script-src 'self'", security.policy(source))
        self.assertNotIn(security.digest(""), security.policy(source))

    def test_release_requires_security_validation_and_pinned_actions(self):
        workflows = ROOT / ".github/workflows"
        publish = (workflows / "publish-pypi.yml").read_text()
        self.assertIn("needs: validate", publish)
        self.assertIn("uses: ./.github/workflows/security-audit.yml", publish)
        self.assertIn("workflow_call:", (workflows / "security-audit.yml").read_text())
        for workflow in workflows.glob("*.yml"):
            for action in re.findall(r"uses:\s+(\S+)", workflow.read_text()):
                self.assertTrue(
                    action.startswith("./")
                    or re.fullmatch(r"[^@]+@[0-9a-f]{40}", action),
                    action,
                )

    def test_docker_runs_unprivileged_and_copies_only_runtime_inputs(self):
        docker = (ROOT / "Dockerfile").read_text()
        self.assertIn("USER aag", docker)
        self.assertNotIn("COPY python/ /", docker)
        self.assertIn("python/LICENSE.md", docker)
        ignored = (ROOT / ".dockerignore").read_text().splitlines()
        for pattern in [
            "**/.env",
            "**/.env.*",
            "**/.security-tools",
            "**/node_modules",
            "**/security-reports",
        ]:
            self.assertIn(pattern, ignored)

    def test_interactive_code_window_does_not_move_during_pointer_activation(self):
        css = (ROOT / "docs/website/assets/home.css").read_text()
        self.assertRegex(css, r"\.js \.code-window\.rv\s*\{[^}]*transform: none;")
