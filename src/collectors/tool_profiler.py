"""Tool profiling utilities for capturing code and library versions."""

import importlib.metadata
import logging
import os
from typing import Any

import git

logger = logging.getLogger(__name__)


def get_tool_metadata(repo_path: str | None = None) -> dict[str, Any]:
    """Capture the version of the code and critical libraries used in this run."""
    meta: dict[str, Any] = {}

    # 1. Automatic Repo Path Detection
    # Go up 3 levels from: src/collectors/tool_profiler.py -> project_root/
    if repo_path is None:
        repo_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # 2. Get Git Commit Hash (The "Version" of your pipeline code)
    try:
        repo = git.Repo(repo_path)
        meta["eval_tool_commit_hash"] = repo.head.object.hexsha
        meta["eval_tool_branch"] = repo.active_branch.name
    except Exception:
        # This often happens inside Docker if .git folder isn't copied.
        # That's okay, we just log it.
        meta["git_error"] = "Git metadata unavailable (no .git folder found)"

    # 3. Get Python Package Versions
    # Updated to track 'lighteval' since that is what tasks.py uses
    packages_to_track = ["lighteval", "torch", "transformers", "accelerate", "huggingface_hub"]

    package_versions: dict[str, str] = {}
    for pkg in packages_to_track:
        try:
            package_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            package_versions[pkg] = "Not Installed"

    meta["python_package_versions"] = package_versions

    # 4. Main Tool Version (LightEval)
    meta["eval_tool_version"] = package_versions.get("lighteval", "Unknown")

    return meta
