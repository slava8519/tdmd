# tdmd

Текущий активный цикл разработки: GPU-портируемость (NVIDIA + AMD) через Kokkos.  
План работ и строгие гейты: `docs/PORTABILITY_KOKKOS_PLAN.md` (`PR-K01..PR-K10`).

Публикация в GitLab:
- CI для GitLab: `.gitlab-ci.yml`
- чеклист публикации: `docs/GITLAB_PUBLISH.md`

v1.9 завершает логическую цепочку TD-декомпозиции по времени:

## Что нового

### 1) Связь buffer/skin с time-lag
Теперь `skin` гарантирует корректность не только для перестроения таблиц,
но и для **асинхронного временного лага зон**:

Это означает:
- если зона отстаёт по времени на `L` шагов,
- атом физически не может «перепрыгнуть» через границу зоны и
  потеряться из таблиц взаимодействия.

### 2) Формальная замкнутость TD-схемы
Начиная с v1.9, корректность обеспечивается **тремя независимыми уровнями**:
1. *Автомат состояний* (≤1 зона в W, владение, outbox),
2. *Временная логика* (step_id, max_step_lag),
3. *Физическая достаточность* (buffer/skin ≥ v·dt·lag).

Это ровно та точка, где TD-схема становится
**математически и физически самосогласованной**.

## Конфиг
Ничего нового добавлять не нужно:
- `max_step_lag` автоматически учитывается при вычислении `skin`.

## Запуск
```bash
python -m tdmd.main verify examples/td_1d_morse.yaml --steps 50
mpirun -np 4 python -m tdmd.main run examples/td_1d_morse.yaml --mode td_full_mpi
```


## v1.9 (добавлено)
- Буфер/skin теперь масштабируются с `max_step_lag`: `buffer ~ vmax*dt*(L+1)`.
- В лог добавлен `bufV` — случаи, когда выбранный коэффициент буфера недостаточен для линейной оценки дрейфа.


## v2.0 (добавлено)
- Унифицирован протокол сообщений: computed zones и outbox-атомы передаются в **одном потоке** сообщений через «records».
  - `REC_ZONE` — полное состояние зоны (zid, ids, r, v, step_id)
  - `REC_DELTA` — добавка атомов к зоне (zid, ids, r, v), без изменения step_id зоны
- Убраны дополнительные теги/каналы для outbox: теперь вся связь — один tag_base.


## v2.1 (добавлено) — строгая доставка DELTA по времени зоны
- `REC_DELTA` теперь несёт **целевой step_id** (zone-time), т.е. атомы относятся к конкретному временному слою зоны.
- На приёме действует политика:
  - если зона уже на этом `step_id` → DELTA применяется сразу,
  - если зона ещё не на этом `step_id` → DELTA буферизуется до момента, когда зона достигнет этого слоя,
  - если зона ушла вперёд и DELTA «слишком стара» (`zone.step_id - delta.step_id > max_step_lag`) → DELTA отбрасывается (счётчик `dX`).
- В логе добавлены:
  - `dA` — applied atoms,
  - `dD` — deferred atoms,
  - `dX` — dropped atoms.


## v2.2 (добавлено) — типы DELTA и ограниченная память pending-бфера
- `REC_DELTA` получил поле `subtype` (пока используется `DELTA_MIGRATION=0`, но формат готов для HALO/CORRECTION).
- `pending_deltas` стал **bounded**:
  - автоматически выбрасывает «слишком старые» DELTA-группы по правилу `zone.step_id - delta.step_id > max_step_lag`,
  - ограничивает суммарное число отложенных атомов параметром `td.max_pending_delta_atoms`
    (при переполнении выбрасывает самые старые группы по `step_id`).
- В логе добавлен `dO` — число атомов, отброшенных по переполнению pending-буфера.


## v2.3 (добавлено) — HALO/ghost как отдельный DELTA-subtype, связанный с support(table)
- Добавлен `DELTA_HALO=1`.
- При отправке вычисленной зоны `z`:
  - атомы, которые **нельзя передавать как владение** (overlap = `A(z) ∩ support(table(next))`),
    теперь отправляются в `next` как `REC_DELTA(DELTA_HALO, next_zid, halo_ids, step_id=z.step_id)`.
  - владение остаётся у исходной зоны (эти атомы в `keep_ids`), но их копия доступна следующей зоне для построения таблиц/сил.
- На приёме HALO не смешивается с `atom_ids`, а хранится в `zone.halo_ids` и участвует только как кандидаты в `ensure_table`.
- В лог добавлены счётчики: `hA/hD/hX` (applied/deferred/dropped halo atoms).


## v2.4 (добавлено) — HALO однослойный (по step_id) + уникализация
- `ZoneRuntime` получил `halo_step_id`.
- HALO теперь **однослойный**: при первом применении HALO на новом `step_id` старая `halo_ids` замещается,
  а не накапливается между временными слоями.
- При добавлении HALO выполняется `np.unique(...)`, чтобы:
  - убрать дубли (boundedness),
  - сделать таблицы взаимодействий стабильнее.
- При построении таблиц (`ensure_table`) halo учитывается только если `halo_step_id == step_id` соответствующей зоны-зависимости.


## v2.5 (добавлено) — геометрически ограниченный HALO (p-окрестность) + инвариант
- При отправке `DELTA_HALO` теперь отсылаются **только** те overlap-атомы, чьи координаты попадают в p-окрестность зоны-получателя:
  `[z0(next)-cutoff, z1(next)+cutoff]` (с учётом PBC).
  Это предотвращает «заливку» получателя лишними ghosts.
- Добавлена проверка-инвариант на приёме: halo-атомы должны лежать внутри p-окрестности своей зоны.
  Нарушения считаются в `hG` (halo geometry violations) в логе.


## v2.6 (добавлено) — HALO как таблично-опорный объект (support-driven)
- Генерация HALO теперь определяется **исключительно** множеством overlap:
  `HALO = A(sender) ∩ support(table(receiver))`, а не «все keep_ids».
  Это делает HALO минимально достаточным по смыслу TD-таблиц (диссертационное правило).
- HALO по-прежнему геометрически фильтруется по p-окрестности зоны-получателя.
- Добавлены счётчики:
  - `hS` — сколько halo-атомов отправлено,
  - `hV` — нарушения инварианта `halo ⊆ support(table(zone))` (проверяется, если table уже построена на приёме).


## v2.7 (добавлено) — двусторонний HALO по зависимостям (bidirectional)
- Реализован **двунаправленный** обмен HALO:
  - forward-поток (как раньше) в `next_rank` на `tag_base`,
  - backward-поток HALO в `prev_rank` на `tag_base+2`.
- Для вычисленной зоны `z` формируются HALO для двух ближайших зависимостей (1D-прототип):
  - `next_zid = z+1` (forward),
  - `prev_zid = z-1` (backward).
- HALO остаётся support-driven и геометрически ограниченным p-окрестностью получателя.
- В лог добавлены счётчики:
  - `hF` — отправлено HALO вперёд,
  - `hB` — отправлено HALO назад.


## v2.8 (добавлено) — HALO по полному списку зависимостей deps(z) (в пределах cutoff)
- HALO теперь формируется **для всех зон deps(z)**, определённых как зоны, пересекающие p-расширение `[z0-cutoff, z1+cutoff]` (с PBC).
- Для каждой зависимости делается:
  - попытка support-driven halo через `overlap_set_for_send(sender, dep)` (если таблица/опора доступна локально),
  - fallback на геометрический halo из `A(sender)` с последующим p-фильтром получателя.
- Доставка HALO выполняется в 1D-прототипе через два потока:
  - зависимости “вперёд” отправляются в `next_rank`,
  - зависимости “назад” — в `prev_rank` (tag_base+2).


## v2.9 (добавлено) — адресная маршрутизация HALO/DELTA по текущему держателю зоны (dynamic holder routing)
- Введена динамическая таблица `holder_map[zid] -> rank`, где находится зона `zid` (не-F) **в данный момент**.
  - Инициализация: `allgather` по фактическим зонам на старте.
  - Обновление: при получении `REC_ZONE` и при отправке `REC_ZONE` в следующий ранг.
- HALO и `DELTA_MIGRATION` теперь отправляются **напрямую** рангу-держателю соответствующей зоны-зависимости/назначения.
- Приём стал универсальным: `recv_phase` слушает `MPI.ANY_SOURCE` по одному `tag_base`, что естественно для direct routing.


## v3.0 (добавлено) — формальная готовность по deps + «gossip» holder-map (версионирование)
- Добавлен `REC_HOLDER` (record type=2): короткое сообщение-обновление держателя зоны.
  - Несёт `zid` и `version` (в поле `step_id`).
  - На приёме обновляет `holder_map[zid]` только если `version` новее локальной.
- Введены:
  - `holder_ver[zid]` — версия информации о держателе,
  - локальный счётчик `holder_epoch`, увеличивается при смене держателя.
- Введена **формальная готовность зоны к вычислению**:
  - `td.require_local_deps: true` (по умолчанию)
  - зона допускается в `W` только если все её зависимости `deps(z)` (кроме пустых/F) *локальны* (`holder_map[dep]==rank`).
  - Это даёт строгую управляемость автомата: вычисление не стартует на неполном наборе зависимостей.
- `td.holder_gossip: true` — обновления `REC_HOLDER` подмешиваются (piggyback) к пересылке `REC_ZONE`, ускоряя согласование `holder_map`.


## v3.1 (добавлено) — явное ожидание deps + протокол REQ_HALO (pull-механизм)
- Введён `REC_REQ` (record type=3) и `REQ_HALO=0`: запрос «пришли HALO для зоны-зависимости».
  - Формат: `REC_REQ(zid=dep_zid, ids=[requester_zid, requester_step], step_id=-1)`.
- При получении `REC_REQ` держатель `dep_zid` формирует `DELTA_HALO(dep_zid)` и отправляет прямо запрашивающему рангу.
  - HALO фильтруется по p-окрестности *requester_zid* (если она известна из метаданных).
- Если на ранге пришёл `DELTA_HALO` для зоны `dep_zid`, которой локально не было (она была `F`),
  зона поднимается в состояние `P` как **shadow dependency** (без владения), с `step_id` из сообщения.
- `td.require_local_deps` теперь означает: deps должны быть **локально представлены** (владение или shadow/halo), а не обязательно локально владеться.
- Добавлен proactive-pull: если вычисление блокируется отсутствием deps, ранк посылает `REQ_HALO` держателям недостающих зависимостей.
- В лог добавлены: `reqS/reqR` (sent/received requests), `sh` (promoted shadow zones).


## v3.2 (добавлено) — два класса зависимостей: table-deps и owner-deps
- Введены два независимых класса deps:
  1) **table-deps** — зависимости, необходимые для построения таблиц/сил (HALO/ghost достаточно).
  2) **owner-deps** — зависимости, необходимые для более сильных инвариантов владения (опционально, по умолчанию выключено).
- Новые параметры:
  - `td.require_table_deps` (default true): зона допускается в `W` только если все `deps_table(z)` *представлены локально* (`ztype != F`, т.е. owned или shadow/halo).
  - `td.require_owner_deps` (default false): дополнительное ограничение по owner-deps (прототип).
  - `td.require_local_deps` оставлен как legacy alias для `require_table_deps`.
- Автомат теперь считает, что лаг/готовность проверяются относительно **table-deps**, а owner-deps — отдельным предикатом.
- В лог добавлены:
  - `wT` — сколько раз вычисление блокировалось по отсутствию table-deps,
  - `wO` — сколько раз блокировалось по owner-deps.


## v3.3 (добавлено) — формальное определение owner-deps через миграционный радиус (buffer)
- `owner-deps(z)` определены геометрически как зоны, пересекающие диапазон возможной миграции атомов за шаг:
  `[z0(z)-buffer(z), z1(z)+buffer(z)]` (с PBC).
  Здесь `buffer(z)` уже учитывает `max_step_lag` (см. v1.9), т.е. это корректный worst-case в TD.
- `td.require_owner_deps` теперь проверяет, что для всех `owner-deps` **известен держатель** (`holder_map[dep] != -1`).
  Это минимально необходимый инвариант для строгой маршрутизации миграций (не отправлять в «никуда»).
- В лог добавлен счётчик `wOU` — случаи, когда прогресс блокируется/ограничивается неизвестными держателями owner-deps.


## v3.4 (добавлено) — pull для holder-map: REQ_HOLDER + проверка свежести holder_ver для owner-deps
- Добавлен `REQ_HOLDER=1` (подтип `REC_REQ`): запрос актуальной информации о держателе зоны `zid`.
  - Держатель/получатель отвечает записью `REC_HOLDER(zid, version=holder_ver[zid])`.
- Для `td.require_owner_deps=true` добавлено требование свежести (если `td.require_owner_ver=true`):
  - `holder_ver[dep] >= holder_epoch - (2*max_step_lag + 2)`
  - т.е. сведения о держателе должны быть *не старее* окна TD-лага.
- В proactive-pull при блокировке:
  - если owner-deps неизвестен или сведения stale, посылается `REQ_HOLDER` (если `enable_req_holder=true`).
- В лог добавлены:
  - `reqHS/reqHR` — отправленные/обработанные запросы держателя,
  - `wOS` — блокировки из-за stale holder-информации.


## verify2 — тестовый стенд (научная верификация)
Команда сравнивает `serial` vs `td_local` не только по `r/v`, но и по базовым термодинамическим величинам:

- энергия: `E = KE + PE`,
- температура: `T = 2KE/(3N)` (kB=1),
- давление по вириалу: `P = N*T/V + W/(3V)`.

Запуск:
```
python -m tdmd.main verify2 examples/td_1d_morse.yaml --steps 200 --every 10
```

Доступные фиксированные кейсы:
- `gas_lowrho_lj`
- `fcc_solid`
- `bcc_solid`

Можно выбрать подмножество:
```
python -m tdmd.main verify2 examples/td_1d_morse.yaml --case fcc_solid --case bcc_solid
```


## verifylab — регрессионная лаборатория (матрица параметров + CSV + графики)
Команда запускает `verify2` на фиксированном наборе кейсов в *свипе параметров* и пишет результат в CSV.

Пример:
```
python -m tdmd.main verifylab examples/td_1d_morse.yaml --steps 200 --every 10 \
  --zones_total 4 --zones_total 8 --zones_total 16 \
  --use_verlet true --use_verlet false \
  --verlet_k_steps 5 --verlet_k_steps 10 \
  --csv out/verifylab.csv --plots out/plots
```

В CSV для каждой комбинации и кейса пишутся:
`max_dr,max_dv,max_dE,max_dT,max_dP` и флаг `ok`.

Если указать `--plots DIR`, будут построены PNG-графики метрик (лог-ось) по индексу свипа, по одному графику на метрику.


## Golden trajectories (эталонные ряды) — publishable verification
Добавлены команды:

Сгенерировать эталон (serial) для фиксированных кейсов:
```
python -m tdmd.main golden-gen examples/td_1d_morse.yaml --out tests/golden/default_golden.json --steps 50 --every 10
```

Проверить `td_local` против эталона (без повторного прогона serial):
```
python -m tdmd.main golden-check examples/td_1d_morse.yaml --golden tests/golden/default_golden.json --steps 50 --every 10
```

Эталон хранит ряды `E,T,P` по шагам наблюдения для каждого кейса.


## Chaos / асинхронный стресс-тест (v3.8)
`verifylab` теперь умеет ось «хаоса»: рандомизация расписания и вероятностная задержка применения DELTA/HALO.

Пример:
```
python -m tdmd.main verifylab examples/td_1d_morse.yaml --steps 200 --every 10 \
  --zones_total 8 --use_verlet true --verlet_k_steps 10 \
  --chaos true --chaos_delay_prob 0.05 --chaos_seed 123 \
  --csv out/verifylab_chaos.csv --plots out/plots_chaos
```

## Доказательная часть и масштабирование deps-графа
См. `docs/td_proofs_and_scaling.md` — список инвариантов (I1–I6), лемм и план перехода от 1D к общему deps-графу.


## 3D zone decomposition (v3.9)

Помимо классического 1D-разбиения по оси Z, добавлен экспериментальный режим **3D-блоков**.
Он предназначен как базис для обобщения TD на произвольный граф зависимостей (не только 1D/кольцо)
и сейчас поддерживается в **td_local** и верификационных сценариях.

В YAML-конфиге:

```yaml
td:
  decomposition: "3d"   # "1d" (default) | "3d"
  zones_nx: 2
  zones_ny: 2
  zones_nz: 2
  zones_total: 8        # должен равняться zones_nx*zones_ny*zones_nz
```

Примечание: **td_full_mpi** остаётся 1D/кольцом; 3D-декомпозиция для MPI требует отдельной схемы обменов по графу.


## v4.1
- `static_3d` интегрирован в TD-MPI: deps по 3D блокам + адресная маршрутизация owner_rank.
- HALO/overlap фильтрация для `static_3d` выполняется по 3D AABB (с PBC), а не по 1D интервалам.


## v4.2
- `static_3d`: таблицы взаимодействий в W теперь могут строиться как локальный 3D CellList (через `InteractionTableState.impl = CellList`).
- `TDAutomaton1W.ensure_table` поддерживает `geom_aabb` и/или `geom_provider` для построения 3D-таблиц.
- В TD-MPI: `overlap_mode=geometric_rc` для `static_3d` использует AABB-overlap вместо 1D zonecells.


## v4.3
- `InteractionTableState` хранит 3D AABB-support (`lo/hi`) для `static_3d`.
- Добавлены проверяемые инварианты 3D: `hG3` (halo geo), `hV3` (halo in support), `tG3` (table candidate geo).
- Введён модуль `geom_pbc.py` для AABB-проверок с периодикой.


## Codex-friendly development
- See `AGENTS.md` for agent roles.
- Quick checks:
  - `python -m pytest -q`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`

## Universal visualization workflow (v4.9)

Visualization/export is a passive observability layer and works across `serial`, `td_local`, and `td_full_mpi`.

Run with trajectory + metrics + schema manifests:
```bash
python -m tdmd.main run examples/td_1d_morse.yaml \
  --task examples/interop/task.yaml \
  --mode td_local --device cpu \
  --traj results/viz_demo/traj.lammpstrj.gz --traj-every 10 \
  --traj-channels unwrapped,image --traj-compression gz \
  --metrics results/viz_demo/metrics.csv --metrics-every 10
```

Post-process trajectory with built-in plugins:
```bash
python scripts/viz_analyze.py \
  --traj results/viz_demo/traj.lammpstrj.gz \
  --plugin mobility \
  --plugin species_mixing \
  --out-csv results/viz_demo/viz/analysis.csv \
  --out-json results/viz_demo/viz/analysis.json
```

Prepare OVITO/VMD/ParaView bundle artifacts:
```bash
python scripts/viz_bundle.py \
  --traj results/viz_demo/traj.lammpstrj.gz \
  --outdir results/viz_demo/viz
```

Strict visualization contract gate:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset viz_smoke --strict
```

## Open EAM library (metals)

Bundled open `eam/alloy` files are located in:
- `examples/potentials/eam_alloy`

Provenance and citation metadata:
- `examples/potentials/README.md`
- `examples/potentials/eam_alloy/README.md`
- `examples/potentials/eam_alloy/library.json`
- `examples/potentials/eam_alloy/SHA256SUMS`

Al-Cu crack tasks are switched to real EAM potential:
- `examples/interop/task_al_cu_crack_10k_nvt.yaml`
- `examples/interop/task_al_cu_crack_10k_nvt_smoke20.yaml`
- `examples/interop/task_al_cu_crack_100k_nvt.yaml`
- `examples/interop/task_al_cu_crack_100k_nvt_smoke20.yaml`

Additional single-metal reference tasks:
- `examples/interop/task_eam_fe_mishin2006.yaml`
- `examples/interop/task_eam_ni99.yaml`
- `examples/interop/task_eam_ti_zhou04.yaml`

Generation script now supports `eam/alloy` directly:
```bash
python scripts/generate_al_cu_crack_task.py \
  --out examples/interop/task_al_cu_crack_10k_nvt_smoke20.yaml \
  --target-atoms 10000 --box 53.0 --steps 20 \
  --potential-kind eam/alloy \
  --eam-file examples/potentials/eam_alloy/AlCu.eam.alloy
```


## Codex bootstrap
- Roles: `AGENTS.md`
- Master prompt: `CODEX_MASTER_PROMPT.md`
- Quick check: `make verify-smoke`
