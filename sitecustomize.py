"""Local test runner guardrails.

Disable auto-loading of globally installed pytest plugins so the project's
tests are isolated from unrelated user-environment packages.
"""

import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
