# Story backlog (local) ↔ GitHub issues

**Scope:** files here help **align and sequence work** with GitHub issues (`gh`, sync scripts). They are **not** the source of truth for math, dataset definitions, or benchmark design — use [`docs/README.md`](../docs/README.md) (especially [`docs/math/`](../docs/math/), e.g. [`benchmark-families.md`](../docs/math/benchmark-families.md)) and [`src/`](../src/).

Markdown files in this directory are the **canonical local copy** of backlog stories (issue titles, checklists, frontmatter). There is one file per item in the “Recommended order” list (**`001`–`021`**), matching [`chatgpt_stories.md`](../chatgpt_stories.md). Filenames use **`NNN-kebab-slug.md`** where `NNN` is that position (Phase 1–4), not the epic-grouped order in the main body of that file.

Each file has **YAML frontmatter**:

- `story` — ChatGPT “Story N” id from the epic sections (cross-reference).
- `recommended_order` / `phase` — ordering from “Recommended order”.
- `issue` — GitHub issue number once linked (omit or `null` until created).
- `title` — Issue title for `gh issue create` / `edit`.
- `labels` — Label names to apply on push (must exist in the repo, or create them first).
- `sync.last_remote_updated` / `sync.content_sha256` — Updated by the sync script.

**Stable identity** is the **filename slug** and `issue` number, not the GitHub title text.

## Commands

Requires [`gh`](https://cli.github.com/) authenticated for this repo and **PyYAML** (`pip install -e '.[dev]'` or `pip install pyyaml`).

```bash
# Default: merge local vs GitHub (checkbox counts + checklist size + timestamps)
python3 scripts/sync_github_stories.py sync

# Force overwrite local from GitHub (bootstrap / “take remote”)
python3 scripts/sync_github_stories.py pull

# Push local title/body/labels to GitHub (creates an issue if `issue` is missing)
python3 scripts/sync_github_stories.py push

# Preview actions
python3 scripts/sync_github_stories.py sync --dry-run

# One file or one issue
python3 scripts/sync_github_stories.py sync --only stories/005-upgrade-streamlit-visualizer-to-workbench.md
python3 scripts/sync_github_stories.py sync --issue 12
```

With the project venv (`make install`):

```bash
make stories-sync
```

## Troubleshooting

On failure, the script prints the **`gh` command**, **stdout/stderr**, and **hints** (network, authentication, permissions, missing labels, rate limits).

If you see **`dial tcp` … `network is unreachable`** (or similar), the problem is **connectivity to GitHub** (offline, bad route, VPN, firewall, proxy) — fix DNS/HTTPS first; `gh auth login` will not help until `curl -I https://api.github.com` works.

`issue create` often fails when a **label in frontmatter does not exist** on the repo. Check with `gh label list` and create any missing names, for example:

```bash
gh label create "epic:streamlit" --color "5319E7"
gh label create "streamlit" --color "1D76DB"
```

Then run `push` again (or trim `labels:` to only names that already exist).

## Note on issue numbers

GitHub issue **#10** in this repo is **Story 3** (run manifest), not Story 10. Story 10 lives in `005-upgrade-streamlit-visualizer-to-workbench.md` and does not have an issue number until you run `push` on that file.
