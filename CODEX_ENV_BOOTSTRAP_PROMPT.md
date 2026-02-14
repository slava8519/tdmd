# Codex Environment Bootstrap Prompt

Use this prompt at the start of a Codex run on a fresh machine to install the local tooling needed for TDMD work.

```text
You are Codex working in the TDMD repository.

Goal: bootstrap this machine for productive code work in this repo.

Do exactly this:
1) Verify repo state:
   - `git status --porcelain=v1`
   - `git branch --show-current`

2) Install required system tools (Ubuntu/Debian):
   - `sudo apt-get update`
   - `sudo apt-get install -y python3.12-venv openmpi-bin libopenmpi-dev ripgrep fd-find jq yq bat git-delta shellcheck hyperfine ncdu`

3) Prepare Python environment:
   - Create `.venv` if missing: `python3 -m venv .venv`
   - Install dependencies from project metadata: `.venv/bin/python -m pip install --upgrade pip`
   - Install project + dev extras: `.venv/bin/python -m pip install -e '.[dev]'`

4) Configure shell aliases in `~/.bashrc` (idempotent block):
   - `alias fd='fdfind'`
   - `alias bat='batcat'`

5) Validate tooling:
   - `rg --version`
   - `fdfind --version`
   - `jq --version`
   - `yq --version`
   - `batcat --version`
   - `delta --version`
   - `shellcheck --version`
   - `hyperfine --version`
   - `ncdu --version`
   - `.venv/bin/python -m pytest -q`

6) Report:
   - what was installed,
   - command outcomes,
   - any blockers and exact failing command.

Prefer running `scripts/bootstrap_codex_env.sh` when present.
```

Quick path:
```bash
bash scripts/bootstrap_codex_env.sh
```
