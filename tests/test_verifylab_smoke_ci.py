from __future__ import annotations
import subprocess
import sys


def test_verifylab_smoke_ci_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "smoke_ci",
        "--strict",
        "--run-id",
        "pytest_smoke_ci",
    ]
    subprocess.check_call(cmd)


def test_verifylab_metal_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "metal_smoke",
        "--strict",
        "--run-id",
        "pytest_metal_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_interop_metal_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "interop_metal_smoke",
        "--strict",
        "--run-id",
        "pytest_interop_metal_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_gpu_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "gpu_smoke",
        "--strict",
        "--run-id",
        "pytest_gpu_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_gpu_interop_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "gpu_interop_smoke",
        "--strict",
        "--run-id",
        "pytest_gpu_interop_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_gpu_metal_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "gpu_metal_smoke",
        "--strict",
        "--run-id",
        "pytest_gpu_metal_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_metal_property_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "metal_property_smoke",
        "--strict",
        "--run-id",
        "pytest_metal_property_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_interop_metal_property_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "interop_metal_property_smoke",
        "--strict",
        "--run-id",
        "pytest_interop_metal_property_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_nvt_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "nvt_smoke",
        "--strict",
        "--run-id",
        "pytest_nvt_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_npt_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "npt_smoke",
        "--strict",
        "--run-id",
        "pytest_npt_smoke",
    ]
    subprocess.check_call(cmd)


def test_verifylab_viz_smoke_strict():
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "viz_smoke",
        "--strict",
        "--run-id",
        "pytest_viz_smoke",
    ]
    subprocess.check_call(cmd)
