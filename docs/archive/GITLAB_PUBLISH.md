# GitLab Publish Guide (`tdmd`)

## 1. Create project
- Create a new empty project in GitLab with name: `tdmd`.
- Do not initialize with README/license/gitignore in GitLab UI.

## 2. Set remote
```bash
git remote remove origin 2>/dev/null || true
git remote add origin git@gitlab.com:<namespace>/tdmd.git
```

HTTPS alternative:
```bash
git remote add origin https://gitlab.com/<namespace>/tdmd.git
```

## 3. First push
```bash
git branch -M main
git push -u origin main
```

## 4. CI expectations
- GitLab CI config is provided in `.gitlab-ci.yml`.
- Default pipeline runs:
  - `python -m pytest -q`
  - `smoke_ci --strict`
  - `interop_smoke --strict`

## 5. Recommended GitLab settings
- Protect `main` branch.
- Require merge requests for `main`.
- Require successful pipeline before merge.
- Enable container/package registry only if needed.

## 6. Post-publish sanity
```bash
git remote -v
python -m pytest -q tests/test_repo_smoke.py
```
