from __future__ import annotations

from scripts import run_verifylab_matrix as cli
from tdmd.verifylab.presets import preset_names
from tdmd.verifylab.runner import build_arg_parser, main as runner_main


def test_verifylab_runner_parser_uses_preset_names():
    parser = build_arg_parser()
    preset_action = next(a for a in parser._actions if getattr(a, "dest", "") == "preset")

    assert tuple(preset_action.choices) == preset_names()
    assert parser.parse_args(["examples/td_1d_morse.yaml"]).preset == "smoke_ci"


def test_verifylab_script_is_thin_cli_wrapper():
    assert cli.main is runner_main
