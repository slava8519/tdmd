from __future__ import annotations
import pytest

from tdmd.io.task import TaskValidationError, load_task, parse_task_dict, validate_task_for_run


def _base_task_dict() -> dict:
    return {
        "task_version": 1,
        "units": "lj",
        "box": {"x": 10.0, "y": 10.0, "z": 10.0, "pbc": [True, True, True]},
        "atoms": [
            {"id": 1, "type": 1, "mass": 1.0, "r": [1.0, 1.0, 1.0], "v": [0.0, 0.0, 0.0]},
            {"id": 2, "type": 1, "mass": 1.0, "r": [2.0, 2.0, 2.0], "v": [0.0, 0.0, 0.0]},
        ],
        "potential": {"kind": "lj", "params": {"epsilon": 1.0, "sigma": 1.0}},
        "cutoff": 2.5,
        "dt": 0.005,
        "steps": 10,
    }


def test_validate_task_for_run_accepts_base_task():
    task = parse_task_dict(_base_task_dict())
    assert task.ensemble.kind == "nve"
    masses = validate_task_for_run(task)
    assert masses.shape == (2,)
    assert masses.tolist() == pytest.approx([1.0, 1.0])


def test_validate_task_for_run_rejects_thermostat():
    d = _base_task_dict()
    d["thermostat"] = {"kind": "berendsen", "params": {"tau": 0.1}}
    task = parse_task_dict(d)
    with pytest.raises(TaskValidationError, match="thermostat"):
        validate_task_for_run(task)


def test_validate_task_for_run_rejects_non_nve_ensemble():
    d = _base_task_dict()
    d["ensemble"] = {
        "kind": "nvt",
        "thermostat": {"kind": "berendsen", "params": {"tau": 0.1, "t_target": 300.0}},
    }
    task = parse_task_dict(d)
    with pytest.raises(TaskValidationError, match="ensemble.kind"):
        validate_task_for_run(task)


def test_validate_task_for_run_accepts_nvt_when_mode_allows():
    d = _base_task_dict()
    d["ensemble"] = {
        "kind": "nvt",
        "thermostat": {"kind": "berendsen", "params": {"tau": 0.1, "t_target": 1.0}},
    }
    task = parse_task_dict(d)
    masses = validate_task_for_run(task, allowed_ensemble_kinds=("nve", "nvt"))
    assert masses.tolist() == pytest.approx([1.0, 1.0])


def test_validate_task_for_run_accepts_npt_when_mode_allows():
    d = _base_task_dict()
    d["ensemble"] = {
        "kind": "npt",
        "thermostat": {"kind": "berendsen", "params": {"tau": 0.1, "t_target": 1.0}},
        "barostat": {"kind": "berendsen", "params": {"tau": 1.0, "p_target": 0.0}},
    }
    task = parse_task_dict(d)
    masses = validate_task_for_run(task, allowed_ensemble_kinds=("nve", "nvt", "npt"))
    assert masses.tolist() == pytest.approx([1.0, 1.0])


def test_parse_task_rejects_nvt_without_thermostat():
    d = _base_task_dict()
    d["ensemble"] = {"kind": "nvt"}
    with pytest.raises(TaskValidationError, match="requires ensemble.thermostat"):
        parse_task_dict(d)


def test_parse_task_rejects_nve_with_controls():
    d = _base_task_dict()
    d["ensemble"] = {
        "kind": "nve",
        "thermostat": {"kind": "berendsen", "params": {"tau": 0.1}},
    }
    with pytest.raises(TaskValidationError, match="kind=nve"):
        parse_task_dict(d)


def test_parse_task_accepts_npt_contract_structure():
    d = _base_task_dict()
    d["ensemble"] = {
        "kind": "npt",
        "thermostat": {"kind": "langevin", "params": {"tau": 0.2, "t_target": 300.0}},
        "barostat": {"kind": "berendsen", "params": {"tau": 1.0, "p_target": 0.0}},
    }
    task = parse_task_dict(d)
    assert task.ensemble.kind == "npt"
    assert task.ensemble.thermostat is not None
    assert task.ensemble.barostat is not None


def test_parse_task_rejects_legacy_thermostat_combined_with_ensemble():
    d = _base_task_dict()
    d["thermostat"] = {"kind": "berendsen", "params": {"tau": 0.1}}
    d["ensemble"] = {
        "kind": "nvt",
        "thermostat": {"kind": "berendsen", "params": {"tau": 0.1}},
    }
    with pytest.raises(TaskValidationError, match="deprecated"):
        parse_task_dict(d)


def test_validate_task_for_run_rejects_non_periodic_boundaries():
    d = _base_task_dict()
    d["box"]["pbc"] = [True, False, True]
    task = parse_task_dict(d)
    with pytest.raises(TaskValidationError, match="periodic boundaries"):
        validate_task_for_run(task)


def test_validate_task_for_run_rejects_charges():
    d = _base_task_dict()
    d["atoms"][0]["charge"] = 0.1
    task = parse_task_dict(d)
    with pytest.raises(TaskValidationError, match="charges are not supported"):
        validate_task_for_run(task)


def test_validate_task_for_run_rejects_unknown_potential_param():
    d = _base_task_dict()
    d["potential"]["params"]["bogus"] = 123
    task = parse_task_dict(d)
    with pytest.raises(TaskValidationError, match="unsupported potential.params"):
        validate_task_for_run(task)


def test_validate_task_for_run_accepts_table_runtime(tmp_path):
    table_path = tmp_path / "pair.table"
    table_path.write_text(
        "# minimal table\n"
        "PAIR\n"
        "N 3 R 0.9 2.5\n"
        "1 0.900000 1.000000 0.000000\n"
        "2 1.500000 0.100000 0.000000\n"
        "3 2.500000 0.000000 0.000000\n",
        encoding="utf-8",
    )
    d = _base_task_dict()
    d["potential"] = {"kind": "table", "params": {"file": str(table_path), "keyword": "PAIR"}}
    task = parse_task_dict(d)
    masses = validate_task_for_run(task)
    assert masses.tolist() == pytest.approx([1.0, 1.0])


def test_validate_task_for_run_warns_on_non_lj_units():
    d = _base_task_dict()
    d["units"] = "metal"
    task = parse_task_dict(d)
    with pytest.warns(RuntimeWarning, match="no unit conversion"):
        masses = validate_task_for_run(task)
    assert masses.tolist() == pytest.approx([1.0, 1.0])


def test_validate_task_for_run_accepts_multi_mass_multi_type_for_pair_runtime():
    d = _base_task_dict()
    d["atoms"] = [
        {"id": 1, "type": 1, "mass": 1.0, "r": [1.0, 1.0, 1.0], "v": [0.0, 0.0, 0.0]},
        {"id": 2, "type": 2, "mass": 2.0, "r": [2.0, 2.0, 2.0], "v": [0.0, 0.0, 0.0]},
    ]
    task = parse_task_dict(d)
    masses = validate_task_for_run(task)
    assert masses.tolist() == pytest.approx([1.0, 2.0])


def test_validate_task_for_run_can_require_uniform_mass():
    d = _base_task_dict()
    d["atoms"] = [
        {"id": 1, "type": 1, "mass": 1.0, "r": [1.0, 1.0, 1.0], "v": [0.0, 0.0, 0.0]},
        {"id": 2, "type": 2, "mass": 2.0, "r": [2.0, 2.0, 2.0], "v": [0.0, 0.0, 0.0]},
    ]
    task = parse_task_dict(d)
    with pytest.raises(TaskValidationError, match="single mass"):
        validate_task_for_run(task, require_uniform_mass=True)


def test_parse_task_rejects_incomplete_pair_coeff_matrix():
    d = _base_task_dict()
    d["atoms"] = [
        {"id": 1, "type": 1, "mass": 1.0, "r": [1.0, 1.0, 1.0], "v": [0.0, 0.0, 0.0]},
        {"id": 2, "type": 2, "mass": 1.0, "r": [2.0, 2.0, 2.0], "v": [0.0, 0.0, 0.0]},
    ]
    d["potential"]["params"] = {
        "pair_coeffs": {
            "1-1": {"epsilon": 1.0, "sigma": 1.0},
            "1-2": {"epsilon": 0.5, "sigma": 1.1},
        }
    }
    with pytest.raises(TaskValidationError, match="does not cover all type pairs"):
        parse_task_dict(d)


def test_validate_task_for_run_accepts_complete_pair_coeff_matrix():
    d = _base_task_dict()
    d["atoms"] = [
        {"id": 1, "type": 1, "mass": 1.0, "r": [1.0, 1.0, 1.0], "v": [0.0, 0.0, 0.0]},
        {"id": 2, "type": 2, "mass": 2.0, "r": [2.0, 2.0, 2.0], "v": [0.0, 0.0, 0.0]},
    ]
    d["potential"]["params"] = {
        "pair_coeffs": {
            "1-1": {"epsilon": 1.0, "sigma": 1.0},
            "1-2": {"epsilon": 0.5, "sigma": 1.1},
            "2-2": {"epsilon": 0.8, "sigma": 0.9},
        }
    }
    task = parse_task_dict(d)
    masses = validate_task_for_run(task)
    assert masses.tolist() == pytest.approx([1.0, 2.0])


def test_validate_task_for_run_rejects_eam_alloy_by_default_mode_gate():
    task = load_task("examples/interop/task_eam_alloy.yaml")
    with pytest.raises(TaskValidationError, match="not supported"):
        validate_task_for_run(task)


def test_validate_task_for_run_accepts_eam_alloy_when_allowed():
    task = load_task("examples/interop/task_eam_alloy.yaml")
    masses = validate_task_for_run(task, allowed_potential_kinds=("eam/alloy",))
    assert masses.tolist() == pytest.approx([26.9815386, 58.6934])


def test_validate_task_for_run_rejects_nonuniform_mass_when_uniform_is_required():
    task = load_task("examples/interop/task_eam_alloy.yaml")
    with pytest.raises(TaskValidationError, match="single mass"):
        validate_task_for_run(
            task,
            require_uniform_mass=True,
            allowed_potential_kinds=("eam/alloy",),
        )


def test_validate_task_for_run_accepts_uniform_mass_when_uniform_is_required():
    task = load_task("examples/interop/task_eam_alloy_uniform_mass.yaml")
    masses = validate_task_for_run(
        task,
        require_uniform_mass=True,
        allowed_potential_kinds=("eam/alloy",),
    )
    assert masses.tolist() == pytest.approx([26.9815386, 26.9815386])
