# Prompt: Risk/Weakness Burndown Run (v2, Historical/Replay)

> Status: `docs/RISK_BURNDOWN_PLAN_V2.md` is already completed.
> Use this prompt only for historical replay/audit runs.
> For active development, use `CODEX_MASTER_PROMPT.md` and active cycle plan `docs/PORTABILITY_KOKKOS_PLAN.md` (Phase H, PR-K01..PR-K10).

Работаем в репозитории:
`<repo_root>`

Обязательно сразу прочитай и строго соблюдай:
- `<repo_root>/AGENTS.md`
- `<repo_root>/docs/RISK_BURNDOWN_PLAN_V2.md`
- `<repo_root>/docs/TODO.md`
- `<repo_root>/docs/SPEC_TD_AUTOMATON.md`
- `<repo_root>/docs/INVARIANTS.md`
- `<repo_root>/docs/THEORY_VS_CODE.md`

Целевая стратегия проекта:
- домен: металлы и сплавы,
- CPU-референс обязателен,
- GPU только как refinement, без изменения TD-семантики.
- активный цикл: GPU-портируемость (NVIDIA + AMD) через Kokkos, по `docs/PORTABILITY_KOKKOS_PLAN.md`.

Ограничения:
- не нарушать TD-состояния `F/D/P/W/S`,
- не вводить глобальные барьеры,
- сохранять инварианты (`hG3/hV3/tG3` и др.),
- любые поведенческие изменения сопровождать тестами и обновлением документации,
- не принимать "ложный green" из-за fallback (если задача про hardware-strict GPU).

Задача на этот запуск:
- Выполни ОДИН PR-sized пункт из `docs/RISK_BURNDOWN_PLAN_V2.md` в критическом порядке (`PR-V2-01`, затем `PR-V2-02`, ...).
- Сначала кратко укажи, какой пункт берешь, и сразу приступай к реализации.

Обязательные проверки перед завершением:
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
- если PR затрагивает GPU:
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
- если PR затрагивает материалы/потенциалы:
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`
- если PR затрагивает MPI overlap/cuda-aware:
  - `.venv/bin/python scripts/bench_mpi_overlap.py --config examples/td_1d_morse_static_rr_smoke4.yaml --n 4 --overlap-list 0,1 --out results/mpi_overlap.csv --md results/mpi_overlap.md`
- если PR затрагивает кластерную валидацию производительности/устойчивости:
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_scale_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_stability_smoke --strict`
- если PR затрагивает materials property-level reference/gates:
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_property_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_property_smoke --strict`

Что обновлять в docs при изменениях:
- `docs/SPEC_TD_AUTOMATON.md`
- `docs/INVARIANTS.md`
- `docs/THEORY_VS_CODE.md`
- `docs/VERIFYLAB.md`
- для GPU: `docs/GPU_BACKEND.md` и `docs/VERIFYLAB_GPU.md`

Формат отчета в конце:
1) Что сделано (кратко)
2) Измененные файлы
3) Результаты команд проверки (с ключевыми цифрами/ошибками)
4) Какие риски из `RISK_BURNDOWN_PLAN_V2` закрыты/снижены
5) Следующий логичный PR-sized пункт
