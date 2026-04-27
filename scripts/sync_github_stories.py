#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync local stories/*.md (YAML frontmatter + Markdown body) with GitHub issues via `gh`.

Commands:
  pull   — For each file with `issue:` set, overwrite local body/title from GitHub (bootstrap / force from remote).
  push   — Push local body/title/labels to GitHub (create issue if `issue` is missing).
  sync   — Bidirectional merge using checkbox counts, checklist length, and timestamps (default).

Requires: `gh` on PATH, authenticated for the repo. Optional dev dep: PyYAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "Install PyYAML: pip install pyyaml (or pip install -e '.[dev]')",
        file=sys.stderr,
    )
    sys.exit(1)

CHECKBOX_LINE = re.compile(r"^\s*-\s*\[([ xX])\]\s*")

REPO_ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR = REPO_ROOT / "stories"


def _gh_cli_repr(gh_args: list[str]) -> str:
    parts: list[str] = []
    for a in gh_args:
        if " " in a or "\n" in a or not a:
            parts.append(repr(a))
        else:
            parts.append(a)
    return "gh " + " ".join(parts)


def _gh_failure_hints(combined: str) -> list[str]:
    """Return short hint lines based on typical `gh` / GitHub API stderr."""
    t = combined.lower()
    hints: list[str] = []
    # Transport / DNS — distinct from auth (often "dial tcp ... connect: network is unreachable")
    if any(
        x in t
        for x in (
            "network is unreachable",
            "no route to host",
            "connection refused",
            "connection reset",
            "connection timed out",
            "i/o timeout",
            "timeout awaiting response headers",
            "temporary failure in name resolution",
            "could not resolve host",
            "dial tcp",
            "econnrefused",
            "etimedout",
            "enetworkunreachable",
            "tls: handshake",
            "certificate",
        )
    ):
        hints.append(
            "Network: your machine cannot complete HTTPS to GitHub (TCP/DNS/proxy). "
            "Check Wi‑Fi/Ethernet, VPN, firewall, and corporate proxy. "
            "Sanity checks: `ping -c1 github.com`, `curl -I https://api.github.com`. "
            "Until those work, `gh` will fail — this is not fixed by `gh auth login`."
        )
    if any(
        x in t
        for x in (
            "401",
            "not logged in",
            "not authenticated",
            "authentication failed",
            "could not authenticate",
            "bad credentials",
        )
    ):
        hints.append(
            "Auth: run `gh auth login` and `gh auth status`. "
            "For private repos or SSO, you may need `gh auth refresh -h github.com -s repo`."
        )
    if "403" in t or "permission" in t or "resource not accessible" in t:
        hints.append(
            "Permissions: the token may lack `repo` scope or org SSO may need authorization "
            "(GitHub → Settings → Applications)."
        )
    if ("404" in t or "not found" in t) and "label" not in t:
        hints.append(
            "Repository: run `gh repo view` from this directory and check `gh repo set-default` " "if you use forks."
        )
    if "label" in t and any(
        x in t
        for x in (
            "could not",
            "not found",
            "does not exist",
            "unknown",
            "failed to resolve",
            "couldn't",
            "invalid",
        )
    ):
        hints.append(
            "Labels: every `--label` must already exist on the repo. "
            'List: `gh label list`. Create: `gh label create "name" --color "ededed"`. '
            "Or remove unknown labels from the story frontmatter and push again."
        )
    if "rate limit" in t:
        hints.append("Rate limit: wait and retry, or authenticate (higher limits for logged-in users).")
    if not hints:
        hints.append("Re-run with `GH_DEBUG=api gh ...` for verbose HTTP traces (same subcommand as above).")
    return hints


def _print_gh_failure(
    gh_args: list[str],
    proc: subprocess.CompletedProcess[str],
    *,
    context: str,
) -> None:
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    combined = f"{out}\n{err}".strip()
    print(f"error: gh failed ({context})", file=sys.stderr)
    print(f"  command: {_gh_cli_repr(gh_args)}", file=sys.stderr)
    print(f"  exit code: {proc.returncode}", file=sys.stderr)
    if out:
        print("  --- stdout ---", file=sys.stderr)
        for line in out.splitlines():
            print(f"  {line}", file=sys.stderr)
    if err:
        print("  --- stderr ---", file=sys.stderr)
        for line in err.splitlines():
            print(f"  {line}", file=sys.stderr)
    if not out and not err:
        print("  (no output from gh; try `gh auth status`)", file=sys.stderr)
    print("  --- hints ---", file=sys.stderr)
    for h in _gh_failure_hints(combined):
        print(f"  • {h}", file=sys.stderr)


def run_gh(
    gh_args: list[str],
    *,
    check: bool = True,
    context: str = "running gh",
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["gh", *gh_args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        _print_gh_failure(gh_args, proc, context=context)
        sys.exit(1)
    return proc


def gh_issue_view_json(issue: int) -> dict[str, Any]:
    r = run_gh(
        ["issue", "view", str(issue), "--json", "title,body,labels,updatedAt"],
        context=f"loading issue #{issue}",
    )
    return json.loads(r.stdout)


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    if body.startswith("\n"):
        body = body[1:]
    return meta, body


def dump_story(meta: dict[str, Any], body: str) -> str:
    head = yaml.safe_dump(
        meta,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()
    body = body.rstrip() + "\n"
    return f"---\n{head}\n---\n\n{body}"


def body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def checkbox_stats(text: str) -> tuple[int, int]:
    """Return (checked_count, total_checkbox_lines)."""
    checked = 0
    total = 0
    for line in text.splitlines():
        m = CHECKBOX_LINE.match(line)
        if not m:
            continue
        total += 1
        if m.group(1).lower() == "x":
            checked += 1
    return checked, total


def list_story_files(only: Path | None) -> list[Path]:
    if only is not None:
        p = only.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return [p]
    if not STORIES_DIR.is_dir():
        return []
    # Only backlog story files: 001-slug.md (exclude README and other docs)
    return sorted(STORIES_DIR.glob("[0-9][0-9][0-9]-*.md"))


def sync_labels(issue: int, desired: list[str], current_names: list[str]) -> None:
    want = set(desired)
    have = set(current_names)
    for label in sorted(want - have):
        run_gh(
            ["issue", "edit", str(issue), "--add-label", label],
            context=f"adding label {label!r} to issue #{issue}",
        )
    for label in sorted(have - want):
        run_gh(
            ["issue", "edit", str(issue), "--remove-label", label],
            context=f"removing label {label!r} from issue #{issue}",
        )


def apply_remote_to_file(path: Path, remote: dict[str, Any], meta: dict[str, Any]) -> None:
    meta = dict(meta)
    meta["title"] = remote["title"]
    meta["labels"] = [x["name"] for x in remote.get("labels") or []]
    if "sync" not in meta or not isinstance(meta["sync"], dict):
        meta["sync"] = {}
    meta["sync"]["last_remote_updated"] = remote["updatedAt"]
    meta["sync"]["content_sha256"] = body_sha256(remote["body"] or "")
    path.write_text(dump_story(meta, remote["body"] or ""), encoding="utf-8")


def cmd_pull(
    paths: list[Path],
    *,
    dry_run: bool,
) -> int:
    for path in paths:
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        issue = meta.get("issue")
        if issue is None:
            print(f"skip (no issue): {path.name}")
            continue
        if dry_run:
            print(f"pull: would update {path.name} from issue #{issue}")
            continue
        remote = gh_issue_view_json(int(issue))
        apply_remote_to_file(path, remote, meta)
        print(f"pulled: {path.name} <= issue #{issue}")
    return 0


def cmd_push(
    paths: list[Path],
    *,
    dry_run: bool,
) -> int:
    for path in paths:
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        title = meta.get("title")
        if not title:
            print(f"error: missing title in {path.name}", file=sys.stderr)
            return 1
        labels = list(meta.get("labels") or [])
        issue = meta.get("issue")

        if issue is None:
            if dry_run:
                print(f"push: would create issue from {path.name}")
                continue
            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tf:
                tf.write(body)
                body_path = tf.name
            try:
                cmd = [
                    "issue",
                    "create",
                    "--title",
                    str(title),
                    "--body-file",
                    body_path,
                ]
                for lb in labels:
                    cmd.extend(["--label", lb])
                r = run_gh(cmd, context=f"creating issue from {path.name}")
                url = r.stdout.strip()
                m = re.search(r"/issues/(\d+)", url)
                if not m:
                    print(
                        f"error: could not parse issue number from: {url}",
                        file=sys.stderr,
                    )
                    return 1
                new_num = int(m.group(1))
                meta["issue"] = new_num
                if "sync" not in meta or not isinstance(meta["sync"], dict):
                    meta["sync"] = {}
                remote = gh_issue_view_json(new_num)
                meta["sync"]["last_remote_updated"] = remote["updatedAt"]
                meta["sync"]["content_sha256"] = body_sha256(body)
                path.write_text(dump_story(meta, body), encoding="utf-8")
                print(f"created issue #{new_num}: {path.name}")
            finally:
                Path(body_path).unlink(missing_ok=True)
            continue

        num = int(issue)
        remote = gh_issue_view_json(num)
        if dry_run:
            if (remote["body"] or "").strip() != body.strip() or remote["title"] != title:
                print(f"push: would edit issue #{num} from {path.name}")
            cur_labels = [x["name"] for x in remote["labels"]]
            if set(cur_labels) != set(labels):
                print(f"push: would sync labels on #{num}: {sorted(set(labels))}")
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tf:
            tf.write(body)
            body_path = tf.name
        try:
            run_gh(
                ["issue", "edit", str(num), "--title", title, "--body-file", body_path],
                context=f"updating issue #{num} from {path.name}",
            )
        finally:
            Path(body_path).unlink(missing_ok=True)

        cur_labels = [x["name"] for x in remote["labels"]]
        if set(cur_labels) != set(labels):
            sync_labels(num, labels, cur_labels)

        remote2 = gh_issue_view_json(num)
        meta = dict(meta)
        if "sync" not in meta or not isinstance(meta["sync"], dict):
            meta["sync"] = {}
        meta["sync"]["last_remote_updated"] = remote2["updatedAt"]
        meta["sync"]["content_sha256"] = body_sha256(body)
        path.write_text(dump_story(meta, body), encoding="utf-8")
        print(f"pushed: {path.name} => issue #{num}")

    return 0


def decide_sync_direction(
    local_body: str,
    remote_body: str,
    remote_updated: str,
    last_synced: str | None,
) -> str:
    """
    Return 'pull', 'push', or 'conflict'.
    """
    lc, lt = checkbox_stats(local_body)
    rc, rt = checkbox_stats(remote_body or "")

    if rc > lc:
        return "pull"
    if lc > rc:
        return "push"
    if rt > lt:
        return "pull"
    if lt > rt:
        return "push"

    if (local_body or "").strip() == (remote_body or "").strip():
        return "push"  # no-op; caller can skip

    # Tie on checkboxes: prefer newer remote if we have timestamps
    if last_synced and remote_updated:
        # ISO strings compare lexicographically for UTC Z format
        if remote_updated > last_synced:
            return "pull"
        if last_synced > remote_updated:
            return "push"

    return "conflict"


def cmd_sync(
    paths: list[Path],
    *,
    dry_run: bool,
    force: str | None,
) -> int:
    for path in paths:
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        issue = meta.get("issue")
        if issue is None:
            print(f"sync: skip (no issue, use push to create): {path.name}")
            continue

        num = int(issue)
        remote = gh_issue_view_json(num)
        sync_meta = meta.get("sync") if isinstance(meta.get("sync"), dict) else {}
        last_synced = sync_meta.get("last_remote_updated")

        direction = decide_sync_direction(body, remote["body"] or "", remote["updatedAt"], last_synced)
        if force == "local":
            direction = "push"
        elif force == "remote":
            direction = "pull"

        if direction == "conflict":
            print(
                f"CONFLICT: {path.name} vs issue #{num} — use --force local or --force remote",
                file=sys.stderr,
            )
            return 1

        if (body or "").strip() == (remote["body"] or "").strip():
            # still sync labels/title from file if needed
            labels = list(meta.get("labels") or [])
            cur_labels = [x["name"] for x in remote["labels"]]
            title = meta.get("title")
            need_label = set(cur_labels) != set(labels)
            need_title = title and title != remote["title"]
            if not need_label and not need_title:
                print(f"sync: unchanged {path.name}")
                continue
            if dry_run:
                if need_title:
                    print(f"sync: would update title on #{num} for {path.name}")
                if need_label:
                    print(f"sync: would sync labels on #{num}")
                continue
            if need_title:
                run_gh(
                    ["issue", "edit", str(num), "--title", title],
                    context=f"updating title on issue #{num} ({path.name})",
                )
            if need_label:
                sync_labels(num, labels, cur_labels)
            remote2 = gh_issue_view_json(num)
            meta = dict(meta)
            if "sync" not in meta or not isinstance(meta["sync"], dict):
                meta["sync"] = {}
            meta["sync"]["last_remote_updated"] = remote2["updatedAt"]
            meta["sync"]["content_sha256"] = body_sha256(body)
            path.write_text(dump_story(meta, body), encoding="utf-8")
            print(f"sync: metadata only {path.name}")
            continue

        if direction == "pull":
            if dry_run:
                print(f"sync: would pull GitHub -> {path.name} (issue #{num})")
            else:
                apply_remote_to_file(path, remote, meta)
                print(f"sync: pulled {path.name} <= issue #{num}")
        else:
            if dry_run:
                print(f"sync: would push {path.name} -> issue #{num}")
            else:
                rc = cmd_push([path], dry_run=False)
                if rc != 0:
                    return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync stories/*.md with GitHub issues.")
    parser.add_argument(
        "command",
        nargs="?",
        default="sync",
        choices=["pull", "push", "sync"],
        help="pull | push | sync (default)",
    )
    parser.add_argument(
        "--only",
        type=Path,
        help="Single story file under stories/",
    )
    parser.add_argument(
        "--issue",
        type=int,
        metavar="N",
        help="Restrict to the file that references this GitHub issue number",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        choices=["local", "remote"],
        help="For sync: always push (local) or pull (remote)",
    )
    args = parser.parse_args()

    try:
        paths = list_story_files(args.only)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.issue is not None:
        filtered: list[Path] = []
        for p in paths:
            meta, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
            if meta.get("issue") == args.issue:
                filtered.append(p)
        if not filtered:
            print(f"No story file linked to issue #{args.issue}", file=sys.stderr)
            return 1
        paths = filtered

    if not paths:
        print(f"No markdown files in {STORIES_DIR}", file=sys.stderr)
        return 1

    if args.command == "pull":
        return cmd_pull(paths, dry_run=args.dry_run)
    if args.command == "push":
        return cmd_push(paths, dry_run=args.dry_run)
    return cmd_sync(paths, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
