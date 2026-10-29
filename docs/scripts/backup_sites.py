"""Create and verify an allowlisted website-source snapshot (standard library only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = (
    "docs/website",
    "docs/agent-leaderboard-site",
    "docs/scripts",
    "docs/browser-tests",
)
SOURCE_FILES = (
    "README.md",
    "python/README.md",
    "USAGE.md",
    "AGENTS.md",
    "LICENSE.md",
    "docs/PRIVACY_REVIEW.md",
    "docs/LAUNCH_REVIEW.md",
    "docs/security-maintenance.md",
    ".github/workflows/website-checks.yml",
    ".github/workflows/security-audit.yml",
    ".github/workflows/publish-pypi.yml",
    "Dockerfile",
    ".dockerignore",
)
EXCLUDED = {
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "test-results",
    "playwright-report",
    ".git",
}


def source_files(root: Path) -> list[Path]:
    candidates = {root / name for name in SOURCE_FILES if (root / name).exists()}
    for name in SOURCE_DIRS:
        directory = root / name
        for path in directory.rglob("*"):
            parts = path.relative_to(root).parts
            if any(
                part in EXCLUDED or part.startswith((".env", "__qa")) for part in parts
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"Refusing a symlink in website sources: {path}")
            if path.is_file():
                candidates.add(path)
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Source escapes repository: {path}")
    return sorted(candidates)


def verify_backup(path: Path) -> int:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != 1:
            raise ValueError("Unsupported backup format")
        expected = {"manifest.json"}
        for record in manifest["files"]:
            name = record["path"]
            if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                raise ValueError(f"Unsafe archive member: {name}")
            if name in expected:
                raise ValueError(f"Duplicate archive member: {name}")
            expected.add(name)
            content = archive.read(name)
            if (
                len(content) != record["size"]
                or hashlib.sha256(content).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Backup checksum mismatch: {name}")
        if (
            len(archive.namelist()) != len(expected)
            or set(archive.namelist()) != expected
        ):
            raise ValueError("Unexpected or duplicate archive members")
        return len(manifest["files"])


def create_backup(root: Path, output: Path) -> int:
    paths = source_files(root)
    if not paths:
        raise ValueError("No website sources found")
    if any(
        output.resolve().is_relative_to((root / name).resolve()) for name in SOURCE_DIRS
    ):
        raise ValueError("Write backups outside the website source directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing snapshots, including on simultaneous runs.
    created = False
    try:
        with output.open("xb") as handle:
            created = True
            manifest = {
                "format": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": [],
            }
            with ZipFile(handle, "w", compression=ZIP_DEFLATED) as archive:
                for path in paths:
                    content = path.read_bytes()
                    name = path.relative_to(root).as_posix()
                    manifest["files"].append(
                        {
                            "path": name,
                            "size": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                    )
                    archive.writestr(name, content)
                archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        return verify_backup(output)
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="New ZIP path; existing snapshots are never overwritten",
    )
    parser.add_argument(
        "--verify", type=Path, help="Verify a snapshot without extracting it"
    )
    args = parser.parse_args()
    if args.verify:
        print(f"Verified {verify_backup(args.verify)} source files in {args.verify}")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or REPO_ROOT / ".cache/site-backups" / f"sites-{stamp}.zip"
    count = create_backup(REPO_ROOT, output)
    print(f"Created and verified {count} source files: {output}")


if __name__ == "__main__":
    main()
