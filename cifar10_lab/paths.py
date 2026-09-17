import os
import platform
from dataclasses import dataclass
from pathlib import Path


APP_DIRECTORY = "cifar10-backbone-lab"
HOME_ENVIRONMENT_VARIABLE = "CIFAR10_LAB_HOME"


@dataclass(frozen=True)
class LabPaths:
    home: Path
    data_dir: Path
    checkpoints_dir: Path
    results_dir: Path

    def create(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self


def _platform_data_home():
    override = os.environ.get(HOME_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser().resolve()

    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (base / APP_DIRECTORY).resolve()


def get_lab_paths(home=None, create=False):
    base = Path(home).expanduser().resolve() if home else _platform_data_home()
    paths = LabPaths(
        home=base,
        data_dir=base / "data",
        checkpoints_dir=base / "checkpoints",
        results_dir=base / "results",
    )
    return paths.create() if create else paths


def _editable_project_root():
    candidate = Path(__file__).resolve().parent.parent
    return candidate if (candidate / "pyproject.toml").exists() else None


def resolve_data_dir(data_root=None):
    if data_root:
        return Path(data_root).expanduser().resolve()

    # Reuse the original project dataset during editable development.
    project_root = _editable_project_root()
    legacy_data = project_root / "Cifar-10 dataset" if project_root else None
    if legacy_data and legacy_data.exists():
        return legacy_data.resolve()
    return get_lab_paths().data_dir


def resolve_checkpoint_dir(weight_dir=None):
    if weight_dir:
        return Path(weight_dir).expanduser().resolve()
    return get_lab_paths().checkpoints_dir


def resolve_results_dir(results_dir=None):
    if results_dir:
        return Path(results_dir).expanduser().resolve()
    return get_lab_paths().results_dir
