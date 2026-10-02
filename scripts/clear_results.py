"""Clear MOBench result artifacts.

By default this script targets the repository-local results directory and runs
as a dry run. Pass --yes to actually delete the result contents.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = validate_results_dir(args.results_dir, args.allow_nonstandard_results_dir)

    if not results_dir.exists():
        print(f"[clear-results] nothing to clear; directory does not exist: {results_dir}")
        return 0
    if not results_dir.is_dir():
        raise NotADirectoryError(f"results path is not a directory: {results_dir}")

    targets = sorted(results_dir.iterdir(), key=lambda path: str(path).lower())
    if not targets and not args.remove_results_dir:
        print(f"[clear-results] results directory is already empty: {results_dir}")
        return 0

    if not args.yes:
        print(f"[clear-results] dry run for: {results_dir}")
        for target in targets:
            print(f"  would remove {target}")
        if args.remove_results_dir:
            print(f"  would remove directory {results_dir}")
        print("[clear-results] pass --yes to delete these files.")
        return 0

    if args.remove_results_dir:
        shutil.rmtree(results_dir)
        print(f"[clear-results] removed directory: {results_dir}")
        return 0

    for target in targets:
        remove_path(target)
        print(f"[clear-results] removed {target}")
    print(f"[clear-results] cleared contents of: {results_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear MOBench result artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Results directory to clear.")
    parser.add_argument("--yes", action="store_true", default=False, help="Actually delete files. Without this flag the script is a dry run.")
    parser.add_argument("--remove-results-dir", action="store_true", default=False, help="Remove the results directory itself instead of only its contents.")
    parser.add_argument(
        "--allow-nonstandard-results-dir",
        action="store_true",
        default=False,
        help="Allow clearing a directory outside the repo-local results directory.",
    )
    return parser.parse_args(argv)


def validate_results_dir(path: Path, allow_nonstandard: bool) -> Path:
    resolved = path.resolve()
    if allow_nonstandard:
        return resolved
    if resolved.name != "results":
        raise ValueError(f"Refusing to clear non-results directory without --allow-nonstandard-results-dir: {resolved}")
    if not is_relative_to(resolved, REPO_ROOT):
        raise ValueError(f"Refusing to clear path outside repo without --allow-nonstandard-results-dir: {resolved}")
    return resolved


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
