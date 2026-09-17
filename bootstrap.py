"""Create a local environment and prepare the project for a first run."""

import argparse
import subprocess
import sys
import venv
from pathlib import Path


MINIMUM_PYTHON = (3, 10)
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIRECTORY = PROJECT_ROOT / ".venv"


def _venv_python():
    directory = "Scripts" if sys.platform == "win32" else "bin"
    executable = "python.exe" if sys.platform == "win32" else "python"
    return VENV_DIRECTORY / directory / executable


def _run(arguments):
    command = [str(part) for part in arguments]
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create .venv, install the project, check the runtime, and download CIFAR-10."
        )
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Install only the command-line dependencies, without Jupyter/plotting extras.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Do not download CIFAR-10 during setup.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if sys.version_info < MINIMUM_PYTHON:
        required = ".".join(map(str, MINIMUM_PYTHON))
        current = ".".join(map(str, sys.version_info[:3]))
        raise SystemExit(f"Python {required}+ is required; found Python {current}.")

    if not VENV_DIRECTORY.exists():
        print(f"Creating virtual environment: {VENV_DIRECTORY}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIRECTORY)
    else:
        print(f"Using existing virtual environment: {VENV_DIRECTORY}")

    python = _venv_python()
    if not python.exists():
        raise SystemExit(f"Virtual environment Python was not found: {python}")

    _run([python, "-m", "pip", "install", "--upgrade", "pip"])
    install_target = ".[notebook]" if not args.minimal else "."
    _run([python, "-m", "pip", "install", install_target])
    _run([python, "-m", "cifar10_lab", "doctor"])
    if not args.skip_data:
        _run([python, "-m", "cifar10_lab", "download-data"])

    activate = (
        ".venv\\Scripts\\Activate.ps1"
        if sys.platform == "win32"
        else "source .venv/bin/activate"
    )
    print("\nSetup complete.")
    print(f"Activate the environment: {activate}")
    print("Run the beginner example: cifar10-lab train --model resnet18 --quick")
    print("Open the foundations notebook: python -m jupyter lab foundations.ipynb")
    print("Then explore all backbones in: main.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
