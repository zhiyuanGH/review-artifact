"""Install/check PC-side Perfetto tooling for MOBench resource monitoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from resource_monitoring import discover_adb_path, ensure_trace_processor


DEFAULT_TOOLS_DIR = REPO_ROOT / ".wfgen_perfetto"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tools_dir = args.tools_dir.resolve()
    setup = ensure_trace_processor(tools_dir)
    phone = check_phone_perfetto(args)
    result = {
        "tools": setup,
        "phone": phone,
        "note": "MOBench uses on-device /system/bin/perfetto; this script installs only PC-side trace_processor tooling.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.check and (not setup.get("available") or not phone.get("perfetto_available")):
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install/check Perfetto Trace Processor tooling for MOBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS_DIR, help="Local directory for Perfetto tooling.")
    parser.add_argument("--adb", type=Path, default=None, help="adb executable. If omitted, ADB env var, common platform-tools path, or PATH is used.")
    parser.add_argument("--adb-serial", default=None, help="Optional adb device serial.")
    parser.add_argument("--check", action="store_true", help="Return non-zero when tools or phone Perfetto are unavailable.")
    return parser.parse_args(argv)


def check_phone_perfetto(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "adb": None,
        "adb_serial": args.adb_serial,
        "perfetto_available": False,
        "trace_dir_writable": False,
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    try:
        adb = discover_adb_path(args.adb)
        info["adb"] = str(adb)
        serial_args = ["-s", args.adb_serial] if args.adb_serial else []
        help_result = subprocess.run(
            [str(adb), *serial_args, "shell", "sh", "-c", "perfetto --help 2>&1 | head -5"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        info["stdout"] = help_result.stdout
        info["stderr"] = help_result.stderr
        info["perfetto_available"] = help_result.returncode == 0 and "Usage: perfetto" in (help_result.stdout + help_result.stderr)
        write_result = subprocess.run(
            [
                str(adb),
                *serial_args,
                "shell",
                "sh",
                "-c",
                "echo ok > /data/misc/perfetto-traces/wfgen_setup_write_test && rm -f /data/misc/perfetto-traces/wfgen_setup_write_test",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        info["trace_dir_writable"] = write_result.returncode == 0
        if write_result.stderr:
            info["trace_dir_stderr"] = write_result.stderr
    except Exception as exc:
        info["error"] = str(exc)
    return info


if __name__ == "__main__":
    raise SystemExit(main())
