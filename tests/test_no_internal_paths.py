"""Shipped files (per ``.release-include``) must not leak internal strings."""
from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

FORBIDDEN = {
    "internal project path": r"/data/projects",
    "internal conda install": r"/data/environments",
    "cluster account": r"--account=pci",
}
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _git_ls_files(*specs: str) -> list[str]:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ROOT, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", *specs],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def _shipped() -> list[str]:
    specs = [
        line.strip()
        for line in (ROOT / ".release-include").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return _git_ls_files(*specs)


def _hits(pattern: str) -> list[str]:
    regex = re.compile(pattern.encode())
    hits = []
    for rel in _shipped():
        path = ROOT / rel
        if path == SELF or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:  # binary
            continue
        for lineno, line in enumerate(data.splitlines(), 1):
            if regex.search(line):
                hits.append(f"{rel}:{lineno}")
    return hits


def test_release_include_selects_files():
    shipped = set(_shipped())
    assert "pyproject.toml" in shipped
    assert not any(p.endswith("CLAUDE.md") for p in shipped)


@pytest.mark.parametrize("label", sorted(FORBIDDEN))
def test_no_internal_strings(label):
    hits = _hits(FORBIDDEN[label])
    assert not hits, f"{label} in shipped files: {hits}"


def test_shipped_markdown_links_resolve_to_shipped_paths():
    """A relative link must name a shipped file or a directory holding one:
    links to missing files and to excluded (internal) paths both fail."""
    shipped = set(_shipped())
    bad = []
    for rel in sorted(p for p in shipped if p.endswith(".md")):
        for target in MD_LINK.findall((ROOT / rel).read_text(errors="replace")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            path = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target))
            under = path.rstrip("/") + "/"
            if path not in shipped and not any(s.startswith(under) for s in shipped):
                bad.append(f"{rel} -> {target}")
    assert not bad, f"shipped markdown links to missing or unshipped paths: {bad}"


def test_no_decimal_megabit_bpg_constant():
    hits = _hits(r"\b8(\.0)?e6\b")
    assert not hits, f"MiB x 8e6 bpg constant in shipped files: {hits}"
