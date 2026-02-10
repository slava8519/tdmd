from __future__ import annotations

from .cli_commands import (
    cmd_golden_check as _cmd_golden_check_impl,
)
from .cli_commands import (
    cmd_golden_gen as _cmd_golden_gen_impl,
)
from .cli_commands import (
    cmd_run as _cmd_run_impl,
)
from .cli_commands import (
    cmd_verify as _cmd_verify_impl,
)
from .cli_commands import (
    cmd_verify2 as _cmd_verify2_impl,
)
from .cli_commands import (
    cmd_verifylab as _cmd_verifylab_impl,
)
from .cli_parser import build_parser
from .serial import run_serial
from .td_full_mpi import run_td_full_mpi_1d
from .td_local import run_td_local


def _cmd_run(args) -> None:
    return _cmd_run_impl(
        args,
        run_serial=run_serial,
        run_td_local=run_td_local,
        run_td_full_mpi_1d=run_td_full_mpi_1d,
    )


def _cmd_verify(args) -> None:
    return _cmd_verify_impl(args, run_serial=run_serial, run_td_local=run_td_local)


def _cmd_golden_gen(args) -> None:
    return _cmd_golden_gen_impl(args)


def _cmd_golden_check(args) -> None:
    return _cmd_golden_check_impl(args)


def _cmd_verifylab(args) -> None:
    return _cmd_verifylab_impl(args)


def _cmd_verify2(args) -> None:
    return _cmd_verify2_impl(args)


def _build_parser():
    return build_parser(
        cmd_run=_cmd_run,
        cmd_verify=_cmd_verify,
        cmd_golden_gen=_cmd_golden_gen,
        cmd_golden_check=_cmd_golden_check,
        cmd_verifylab=_cmd_verifylab,
        cmd_verify2=_cmd_verify2,
    )


def main() -> None:
    p = _build_parser()
    args = p.parse_args()
    func = getattr(args, "func", None)
    if func is None:
        raise SystemExit(f"unsupported cmd: {getattr(args, 'cmd', None)}")
    func(args)


if __name__ == "__main__":
    main()
