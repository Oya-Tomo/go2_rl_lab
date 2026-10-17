"""Install missing optional dependencies before deployment or video recording.

Overview:
    Check the selected extra from the project's pyproject.toml and install only
    missing or incompatible requirements with the current Python interpreter.

Quick Start:
    python scripts/utils/ensure_dependencies.py mujoco

Notes:
    Requires Python 3.12 or newer. The supported extras are mujoco and video.
    The task and RSL-RL packages are not reinstalled.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata, util
from pathlib import Path
import subprocess
import sys
import tomllib

try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:
    # pip includes packaging when it is not installed as a standalone package.
    from pip._vendor.packaging.requirements import Requirement


class OptionalDependencyInstaller:
    """Check and install the requirements for one optional project feature."""

    def __init__(self, extra: str) -> None:
        """Load the selected optional dependency group from project metadata."""
        project_file = Path(__file__).resolve().parents[2] / "source/go2_rl_lab/pyproject.toml"
        with project_file.open("rb") as stream:
            project = tomllib.load(stream)
        self.extra = extra
        self.requirements = project["project"]["optional-dependencies"][extra]

    def missing_requirements(self) -> list[str]:
        """Return requirements with missing modules or incompatible versions."""
        missing = []
        for requirement_text in self.requirements:
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            module = requirement.name.lower().replace("-", "_")
            if module == "pyyaml":
                module = "yaml"
            try:
                installed_version = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                missing.append(requirement_text)
                continue
            if util.find_spec(module) is None or not requirement.specifier.contains(
                installed_version, prereleases=True
            ):
                missing.append(requirement_text)
        return missing

    def ensure(self) -> None:
        """Install unmet requirements using the current interpreter's pip."""
        missing = self.missing_requirements()
        if not missing:
            return
        print(f"[INFO] Installing missing {self.extra} dependencies: {', '.join(missing)}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        importlib.invalidate_caches()
        remaining = self.missing_requirements()
        if remaining:
            raise RuntimeError(f"Optional dependencies are still unavailable: {', '.join(remaining)}")


def main() -> None:
    """Ensure the requested optional dependency group is ready to use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extra", choices=("mujoco", "video"))
    args = parser.parse_args()
    try:
        OptionalDependencyInstaller(args.extra).ensure()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


if __name__ == "__main__":
    main()
