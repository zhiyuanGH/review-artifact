"""Perfetto-backed resource monitoring for MOBench phone execution."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


TRACE_PROCESSOR_URL = "https://get.perfetto.dev/trace_processor"
REMOTE_TRACE_DIR = "/data/misc/perfetto-traces"
REMOTE_CONFIG_DIR = "/data/misc/perfetto-configs"
REMOTE_TRACE_FILENAME = "wfgen_resource_trace.perfetto-trace"
REMOTE_CONFIG_FILENAME = "wfgen_perfetto_config.pbtxt"
REMOTE_CONFIG_PATH = f"{REMOTE_CONFIG_DIR}/{REMOTE_CONFIG_FILENAME}"
REMOTE_TRACE_PATH = f"{REMOTE_TRACE_DIR}/{REMOTE_TRACE_FILENAME}"
REMOTE_TRACE_DIR_CANDIDATES = [
    REMOTE_TRACE_DIR,
    "/data/local/tmp/wfgen-perfetto-traces",
    "/sdcard/Download/wfgen-perfetto-traces",
]
REMOTE_CONFIG_DIR_CANDIDATES = [
    REMOTE_CONFIG_DIR,
    "/data/local/tmp",
    "/sdcard/Download",
]

DEFAULT_FTRACE_EVENTS = [
    "sched/sched_switch",
    "sched/sched_waking",
    "power/cpu_frequency",
    "power/cpu_idle",
    "kgsl/kgsl_gpubusy",
    "kgsl/kgsl_gpu_frequency",
    "kgsl/kgsl_pwrlevel",
    "kgsl/kgsl_pwrstats",
    "kgsl/kgsl_thermal_constraint",
    "gpu_mem/gpu_mem_total",
    "fastrpc/fastrpc_msg",
    "fastrpc/fastrpc_perf_counters",
    "fastrpc/fastrpc_context_alloc",
    "fastrpc/fastrpc_context_complete",
    "fastrpc/fastrpc_dma_alloc",
    "fastrpc/fastrpc_dma_free",
    "fastrpc/fastrpc_dma_map",
    "fastrpc/fastrpc_dma_unmap",
]

DEFAULT_ATRACE_CATEGORIES = [
    "nnapi",
    "hal",
    "aidl",
    "binder_driver",
    "freq",
    "sched",
    "gfx",
    "view",
    "hwui",
]

CSV_FIELDS = [
    "timestamp_epoch_s",
    "timestamp_utc",
    "cpu_util_percent",
    "mem_total_mb",
    "mem_available_mb",
    "mem_used_mb",
    "mem_used_percent",
    "battery_capacity_percent",
    "battery_temp_c",
    "battery_current_ma",
    "battery_voltage_v",
    "thermal_max_c",
    "gpu_busy_percent",
    "gpu_busy_counter",
    "gpu_total_counter",
    "gpu_freq_hz",
    "gpu_load_percent",
    "llama_server_pid",
    "llama_server_rss_mb",
    "llama_cli_pid",
    "llama_cli_rss_mb",
    "sample_error",
]

SUMMARY_METRICS = [
    "cpu_util_percent",
    "mem_used_mb",
    "mem_used_percent",
    "battery_capacity_percent",
    "battery_temp_c",
    "battery_current_ma",
    "battery_voltage_v",
    "thermal_max_c",
    "gpu_busy_percent",
    "gpu_freq_hz",
    "gpu_load_percent",
    "llama_server_rss_mb",
    "llama_cli_rss_mb",
]


@dataclass
class ResourceMonitorConfig:
    adb_path: Path | None
    device_serial: str | None
    output_dir: Path
    tools_dir: Path
    sample_interval_s: float = 1.0
    buffer_mb: int = 128
    required: bool = False
    dry_run: bool = False
    python_executable: str = sys.executable


def discover_adb_path(explicit_path: Path | None = None) -> Path:
    if explicit_path:
        path = explicit_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"adb not found: {path}")
        return path
    env_adb = os.environ.get("ADB")
    if env_adb and Path(env_adb).exists():
        return Path(env_adb).resolve()
    found = shutil.which("adb")
    if found:
        return Path(found).resolve()
    raise FileNotFoundError("adb not found. Pass --adb or set ADB.")


def ensure_trace_processor(tools_dir: Path) -> dict[str, Any]:
    tools_dir = tools_dir.resolve()
    tools_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = tools_dir / "trace_processor"
    info = {
        "tools_dir": str(tools_dir),
        "wrapper": str(wrapper_path),
        "downloaded": False,
        "available": False,
        "error": None,
    }
    try:
        if not wrapper_path.exists():
            urllib.request.urlretrieve(TRACE_PROCESSOR_URL, wrapper_path)
            info["downloaded"] = True
        info["available"] = wrapper_path.exists()
    except Exception as exc:
        info["error"] = str(exc)
    return info


class PerfettoResourceMonitor:
    """Record a phone Perfetto trace and lightweight ADB resource samples."""

    def __init__(self, config: ResourceMonitorConfig):
        self.config = config
        self.output_dir = config.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.capabilities_path = self.output_dir / "perfetto_capabilities.json"
        self.config_path = self.output_dir / "perfetto_config.pbtxt"
        self.trace_path = self.output_dir / "wfgen_resource_trace.perfetto-trace"
        self.samples_path = self.output_dir / "resource_samples.csv"
        self.summary_path = self.output_dir / "resource_summary.json"
        self.by_workflow_path = self.output_dir / "resource_by_workflow.json"
        self.query_path = self.output_dir / "trace_processor_queries.sql"
        self.trace_processor_output_path = self.output_dir / "trace_processor_output.json"
        self._adb_path: Path | None = None
        self._perfetto_pid: str | None = None
        self._sampler_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._sample_lock = threading.Lock()
        self._previous_cpu: tuple[int, int] | None = None
        self._previous_gpu: tuple[int, int] | None = None
        self.remote_trace_dir = REMOTE_TRACE_DIR
        self.remote_config_dir = REMOTE_CONFIG_DIR
        self.remote_trace_path = REMOTE_TRACE_PATH
        self.remote_config_path = REMOTE_CONFIG_PATH
        self.info: dict[str, Any] = {
            "enabled": True,
            "status": "created",
            "resource_dir": str(self.output_dir),
            "artifacts": self.artifact_paths(),
            "errors": [],
        }

    def artifact_paths(self) -> dict[str, str]:
        return {
            "perfetto_config": str(self.config_path),
            "perfetto_capabilities": str(self.capabilities_path),
            "perfetto_trace": str(self.trace_path),
            "resource_samples": str(self.samples_path),
            "resource_summary": str(self.summary_path),
            "resource_by_workflow": str(self.by_workflow_path),
            "trace_processor_queries": str(self.query_path),
            "trace_processor_output": str(self.trace_processor_output_path),
        }

    def start(self) -> dict[str, Any]:
        if self.config.dry_run:
            self.info.update({"status": "skipped", "reason": "dry_run"})
            self.config_path.write_text("# skipped: dry_run\n", encoding="utf-8")
            self.trace_path.write_bytes(b"")
            self._write_trace_processor_queries()
            self._write_samples_csv()
            self._write_json(
                self.trace_processor_output_path,
                {"attempted": False, "available": False, "trace_exists": False, "reason": "dry_run"},
            )
            self._write_json(self.capabilities_path, {"status": "skipped", "reason": "dry_run"})
            self._write_summary_files([], {}, status="skipped", reason="dry_run")
            return self.info

        try:
            self._adb_path = discover_adb_path(self.config.adb_path)
            self.info["adb"] = str(self._adb_path)
            self.info["adb_serial"] = self.config.device_serial
            capabilities = self.collect_capabilities()
            self._write_json(self.capabilities_path, capabilities)
            self.info["capabilities"] = {
                "perfetto": capabilities.get("perfetto", {}).get("available", False),
                "ftrace_events": len(capabilities.get("selected_ftrace_events", [])),
                "atrace_categories": len(capabilities.get("selected_atrace_categories", [])),
            }
            if not capabilities.get("perfetto", {}).get("available"):
                raise RuntimeError("on-device perfetto is not available")
            trace_dir = capabilities.get("trace_dir")
            config_dir = capabilities.get("config_dir")
            if not trace_dir:
                raise RuntimeError(
                    "no writable remote trace directory found: "
                    + ", ".join(REMOTE_TRACE_DIR_CANDIDATES)
                )
            if not config_dir:
                raise RuntimeError(
                    "no writable remote config directory found: "
                    + ", ".join(REMOTE_CONFIG_DIR_CANDIDATES)
                )
            self.remote_trace_dir = str(trace_dir).rstrip("/")
            self.remote_config_dir = str(config_dir).rstrip("/")
            self.remote_trace_path = f"{self.remote_trace_dir}/{REMOTE_TRACE_FILENAME}"
            self.remote_config_path = f"{self.remote_config_dir}/{REMOTE_CONFIG_FILENAME}"
            self.info["remote_trace_path"] = self.remote_trace_path
            self.info["remote_config_path"] = self.remote_config_path

            config_text = build_perfetto_config(
                capabilities.get("selected_ftrace_events", []),
                capabilities.get("selected_atrace_categories", []),
                self.config.buffer_mb,
            )
            self.config_path.write_text(config_text, encoding="utf-8")
            self._write_trace_processor_queries()
            self._adb_run(["push", str(self.config_path), self.remote_config_path], timeout=30)
            self._adb_shell(f"rm -f {shell_quote(self.remote_trace_path)}", timeout=30, check=False)
            self._start_perfetto()
            self._start_sampler()
            self.info["status"] = "running"
            self.info["started_at_utc"] = utc_now()
            print(f"[resource] Perfetto monitor started; pid={self._perfetto_pid} dir={self.output_dir}", flush=True)
        except Exception as exc:
            self._record_error(exc)
            self.info["status"] = "failed_to_start"
            self._write_summary_files([], {}, status="failed_to_start", reason=str(exc))
            print(f"[resource] monitor start failed: {exc}", flush=True)
            if self.config.required:
                raise
        return self.info

    def stop(self, run_records: list[dict[str, Any]]) -> dict[str, Any]:
        if self.info.get("status") in {"skipped", "failed_to_start"}:
            return self.info
        self.info["stopping_at_utc"] = utc_now()
        self._stop_sampler()
        with self._sample_lock:
            self._samples.append(self._collect_sample())
        self._stop_perfetto()
        self._pull_trace()
        self._write_samples_csv()
        summary = summarize_samples(self._samples)
        by_workflow = summarize_by_workflow(self._samples, run_records)
        self._write_summary_files(self._samples, by_workflow, status="completed", summary=summary)
        self._run_trace_processor()
        self.info["status"] = "completed"
        self.info["finished_at_utc"] = utc_now()
        self.info["sample_count"] = len(self._samples)
        self.info["resource_by_workflow"] = by_workflow
        self.info["artifacts"] = self.artifact_paths()
        print(
            f"[resource] monitor stopped; samples={len(self._samples)} trace={self.trace_path}",
            flush=True,
        )
        return self.info

    def collect_capabilities(self) -> dict[str, Any]:
        perfetto = self._probe_perfetto()
        ftrace_events = self._available_ftrace_events(DEFAULT_FTRACE_EVENTS)
        atrace_categories = self._available_atrace_categories(DEFAULT_ATRACE_CATEGORIES)
        sysfs_paths = self._available_sysfs_paths()
        props = self._device_props()
        trace_dir_probe = self._select_trace_dir()
        config_dir_probe = self._select_writable_dir(REMOTE_CONFIG_DIR_CANDIDATES)
        return {
            "collected_at_utc": utc_now(),
            "device": props,
            "perfetto": perfetto,
            "trace_dir": trace_dir_probe.get("path"),
            "trace_dir_writable": bool(trace_dir_probe.get("writable")),
            "trace_dir_probe": trace_dir_probe,
            "config_dir": config_dir_probe.get("path"),
            "config_dir_writable": bool(config_dir_probe.get("writable")),
            "config_dir_probe": config_dir_probe,
            "requested_ftrace_events": DEFAULT_FTRACE_EVENTS,
            "selected_ftrace_events": ftrace_events,
            "skipped_ftrace_events": [event for event in DEFAULT_FTRACE_EVENTS if event not in ftrace_events],
            "requested_atrace_categories": DEFAULT_ATRACE_CATEGORIES,
            "selected_atrace_categories": atrace_categories,
            "skipped_atrace_categories": [cat for cat in DEFAULT_ATRACE_CATEGORIES if cat not in atrace_categories],
            "sampler_sysfs_paths": sysfs_paths,
            "npu_note": (
                "Perfetto does not expose true HTP/NPU utilization on this build; "
                "NPU visibility is NNAPI/FastRPC/CDSP-adjacent."
            ),
        }

    def _probe_perfetto(self) -> dict[str, Any]:
        result = self._adb_shell("perfetto --help 2>&1 | head -5", timeout=10, check=False)
        combined = result.stdout + result.stderr
        return {
            "available": result.returncode == 0 and "Usage: perfetto" in combined,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def _select_trace_dir(self) -> dict[str, Any]:
        results = []
        for directory in REMOTE_TRACE_DIR_CANDIDATES:
            clean_dir = directory.rstrip("/")
            if clean_dir == REMOTE_TRACE_DIR:
                result = self._adb_shell(f"[ -d {shell_quote(clean_dir)} ] && echo yes", timeout=10, check=False)
                candidate = {
                    "path": clean_dir,
                    "writable": "yes" in result.stdout,
                    "writer": "perfetto",
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "note": (
                        "The shell user may not be able to create files here, but "
                        "on-device perfetto is expected to write traces here."
                    ),
                }
            else:
                candidate = self._probe_shell_writable_dir(clean_dir)
                candidate["writer"] = "shell"
            results.append(candidate)
            if candidate["writable"]:
                return {"path": clean_dir, "writable": True, "candidates": results}
        return {"path": None, "writable": False, "candidates": results}

    def _select_writable_dir(self, candidates: list[str]) -> dict[str, Any]:
        results = []
        for directory in candidates:
            clean_dir = directory.rstrip("/")
            candidate = self._probe_shell_writable_dir(clean_dir)
            results.append(candidate)
            if candidate["writable"]:
                return {"path": clean_dir, "writable": True, "candidates": results}
        return {"path": None, "writable": False, "candidates": results}

    def _probe_shell_writable_dir(self, directory: str) -> dict[str, Any]:
        test_path = f"{directory}/wfgen_write_test_{int(time.time() * 1000)}"
        result = self._adb_shell(
            (
                f"mkdir -p {shell_quote(directory)} 2>/dev/null && "
                f"echo ok > {shell_quote(test_path)} && "
                f"rm -f {shell_quote(test_path)}"
            ),
            timeout=10,
            check=False,
        )
        return {
            "path": directory,
            "writable": result.returncode == 0,
            "writer": "shell",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _available_ftrace_events(self, requested: list[str]) -> list[str]:
        selected = []
        for event in requested:
            command = (
                f"if [ -d /sys/kernel/tracing/events/{event} ] || "
                f"[ -d /sys/kernel/debug/tracing/events/{event} ]; then echo yes; fi"
            )
            result = self._adb_shell(command, timeout=5, check=False)
            if "yes" in result.stdout:
                selected.append(event)
        return selected

    def _available_atrace_categories(self, requested: list[str]) -> list[str]:
        result = self._adb_shell("atrace --list_categories 2>/dev/null", timeout=10, check=False)
        available = set()
        for line in result.stdout.splitlines():
            match = re.match(r"\s*([A-Za-z0-9_]+)\s+-", line)
            if match:
                available.add(match.group(1))
        return [category for category in requested if category in available]

    def _available_sysfs_paths(self) -> list[str]:
        paths = [
            "/sys/class/kgsl/kgsl-3d0/gpubusy",
            "/sys/class/kgsl/kgsl-3d0/gpuclk",
            "/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq",
            "/sys/class/kgsl/kgsl-3d0/devfreq/gpu_load",
            "/sys/class/power_supply/battery/capacity",
            "/sys/class/power_supply/battery/temp",
            "/sys/class/power_supply/battery/current_now",
            "/sys/class/power_supply/battery/voltage_now",
        ]
        command = "for p in " + " ".join(paths) + "; do [ -r \"$p\" ] && echo \"$p\"; done"
        result = self._adb_shell(command, timeout=10, check=False)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _device_props(self) -> dict[str, str]:
        keys = [
            "ro.product.model",
            "ro.product.device",
            "ro.product.board",
            "ro.soc.model",
            "ro.board.platform",
            "ro.build.version.release",
            "ro.build.version.sdk",
        ]
        command = "; ".join(f"echo {key}=$(getprop {key})" for key in keys)
        result = self._adb_shell(command, timeout=10, check=False)
        props = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                props[key.strip()] = value.strip()
        return props

    def _start_perfetto(self) -> None:
        result = self._adb_shell(
            (
                f"perfetto --background-wait --txt -c {shell_quote(self.remote_config_path)} "
                f"-o {shell_quote(self.remote_trace_path)}"
            ),
            timeout=45,
            check=True,
        )
        self.info["perfetto_start_stdout"] = result.stdout
        self.info["perfetto_start_stderr"] = result.stderr
        pid = parse_background_pid(result.stdout)
        if not pid:
            raise RuntimeError(f"Could not parse perfetto background pid from: {result.stdout!r}")
        self._perfetto_pid = pid
        self.info["perfetto_pid"] = pid

    def _stop_perfetto(self) -> None:
        if not self._perfetto_pid:
            return
        result = self._adb_shell(f"kill -TERM {self._perfetto_pid}", timeout=10, check=False)
        self.info["perfetto_stop"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        time.sleep(2.0)

    def _pull_trace(self) -> None:
        if not self._adb_path:
            return
        result = self._adb_run(["pull", self.remote_trace_path, str(self.trace_path)], timeout=120, check=False)
        self.info["trace_pull"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "local_path": str(self.trace_path),
            "exists": self.trace_path.exists(),
            "size_bytes": self.trace_path.stat().st_size if self.trace_path.exists() else 0,
        }

    def _start_sampler(self) -> None:
        self._stop_event.clear()
        self._sampler_thread = threading.Thread(target=self._sample_loop, name="wfgen-resource-sampler", daemon=True)
        self._sampler_thread.start()

    def _stop_sampler(self) -> None:
        self._stop_event.set()
        if self._sampler_thread:
            self._sampler_thread.join(timeout=max(5.0, self.config.sample_interval_s * 3))

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            sample = self._collect_sample()
            with self._sample_lock:
                self._samples.append(sample)
            elapsed = time.time() - started
            wait = max(0.05, self.config.sample_interval_s - elapsed)
            self._stop_event.wait(wait)

    def _collect_sample(self) -> dict[str, Any]:
        now = time.time()
        sample: dict[str, Any] = {
            "timestamp_epoch_s": now,
            "timestamp_utc": datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z"),
            "sample_error": "",
        }
        try:
            result = self._adb_shell(SAMPLE_SCRIPT, timeout=10, check=False)
            if result.returncode != 0:
                sample["sample_error"] = (result.stderr or result.stdout or f"returncode={result.returncode}").strip()
                return sample
            sections = parse_sections(result.stdout)
            self._apply_cpu_sample(sample, sections.get("PROC_STAT", ""))
            self._apply_mem_sample(sample, sections.get("MEMINFO", ""))
            self._apply_battery_sample(sample, sections.get("BATTERY", ""))
            self._apply_gpu_sample(sample, sections.get("GPU", ""))
            self._apply_thermal_sample(sample, sections.get("THERMAL", ""))
            self._apply_process_sample(sample, sections.get("PROCESSES", ""))
        except Exception as exc:
            sample["sample_error"] = str(exc)
        return sample

    def _apply_cpu_sample(self, sample: dict[str, Any], text: str) -> None:
        line = next((item for item in text.splitlines() if item.startswith("cpu ")), "")
        values = [int(part) for part in line.split()[1:] if part.isdigit()]
        if len(values) < 5:
            return
        idle = values[3] + values[4]
        total = sum(values)
        if self._previous_cpu:
            prev_idle, prev_total = self._previous_cpu
            total_delta = total - prev_total
            idle_delta = idle - prev_idle
            if total_delta > 0:
                sample["cpu_util_percent"] = round((1.0 - idle_delta / total_delta) * 100.0, 3)
        self._previous_cpu = (idle, total)

    def _apply_mem_sample(self, sample: dict[str, Any], text: str) -> None:
        values = parse_key_value_kb(text)
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if total:
            sample["mem_total_mb"] = round(total / 1024.0, 3)
        if available is not None:
            sample["mem_available_mb"] = round(available / 1024.0, 3)
        if total and available is not None:
            used = total - available
            sample["mem_used_mb"] = round(used / 1024.0, 3)
            sample["mem_used_percent"] = round(used / total * 100.0, 3)

    def _apply_battery_sample(self, sample: dict[str, Any], text: str) -> None:
        values = parse_equals(text)
        if "capacity" in values:
            sample["battery_capacity_percent"] = parse_float(values["capacity"])
        elif "level" in values:
            sample["battery_capacity_percent"] = parse_float(values["level"])
        if "temp" in values:
            temp = parse_float(values["temp"])
            sample["battery_temp_c"] = round(temp / 10.0, 3) if temp is not None and temp > 100 else temp
        elif "dumpsys_temp" in values:
            temp = parse_float(values["dumpsys_temp"])
            sample["battery_temp_c"] = round(temp / 10.0, 3) if temp is not None else None
        if "current_now" in values:
            current = parse_float(values["current_now"])
            sample["battery_current_ma"] = round(current / 1000.0, 3) if current is not None else None
        if "voltage_now" in values:
            voltage = parse_float(values["voltage_now"])
            sample["battery_voltage_v"] = round(voltage / 1_000_000.0, 3) if voltage is not None else None
        elif "dumpsys_voltage" in values:
            voltage = parse_float(values["dumpsys_voltage"])
            sample["battery_voltage_v"] = round(voltage / 1000.0, 3) if voltage is not None else None

    def _apply_gpu_sample(self, sample: dict[str, Any], text: str) -> None:
        values = parse_equals(text)
        busy_text = values.get("gpubusy", "")
        busy_values = [int(part) for part in re.findall(r"\d+", busy_text)]
        if len(busy_values) >= 2:
            busy, total = busy_values[0], busy_values[1]
            sample["gpu_busy_counter"] = busy
            sample["gpu_total_counter"] = total
            if total > 0:
                sample["gpu_busy_percent"] = round(max(0.0, min(100.0, busy / total * 100.0)), 3)
            self._previous_gpu = (busy, total)
        freq = parse_float(values.get("gpuclk") or values.get("cur_freq"))
        if freq is not None:
            sample["gpu_freq_hz"] = freq
        load = parse_float(values.get("gpu_load"))
        if load is not None:
            sample["gpu_load_percent"] = load

    def _apply_thermal_sample(self, sample: dict[str, Any], text: str) -> None:
        temps = []
        for value in parse_equals(text).values():
            temp = parse_float(value)
            if temp is None:
                continue
            if abs(temp) > 1000:
                temp = temp / 1000.0
            if -40 <= temp <= 150:
                temps.append(temp)
        if temps:
            sample["thermal_max_c"] = round(max(temps), 3)

    def _apply_process_sample(self, sample: dict[str, Any], text: str) -> None:
        for line in text.splitlines():
            values = parse_equals(line.replace(" ", "\n"))
            name = values.get("name")
            if name not in {"llama-server", "llama-cli"}:
                continue
            prefix = "llama_server" if name == "llama-server" else "llama_cli"
            if values.get("pid"):
                sample[f"{prefix}_pid"] = values["pid"]
            rss_kb = parse_float(values.get("rss_kb"))
            if rss_kb is not None:
                sample[f"{prefix}_rss_mb"] = round(rss_kb / 1024.0, 3)

    def _write_samples_csv(self) -> None:
        with self.samples_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for sample in self._samples:
                writer.writerow({field: sample.get(field, "") for field in CSV_FIELDS})

    def _write_summary_files(
        self,
        samples: list[dict[str, Any]],
        by_workflow: dict[str, Any],
        *,
        status: str,
        reason: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        summary_payload = summary or summarize_samples(samples)
        summary_payload.update(
            {
                "status": status,
                "reason": reason,
                "artifacts": self.artifact_paths(),
                "npu_note": (
                    "NPU metrics are NNAPI/FastRPC/CDSP-adjacent and do not represent true HTP utilization percent."
                ),
            }
        )
        self._write_json(self.summary_path, summary_payload)
        self._write_json(self.by_workflow_path, by_workflow)

    def _write_trace_processor_queries(self) -> None:
        self.query_path.write_text(
            "\n".join(
                [
                    (
                        "SELECT 'slice_rows' AS metric, COUNT(*) AS value FROM slice "
                        "UNION ALL SELECT 'counter_rows' AS metric, COUNT(*) AS value FROM counter "
                        "UNION ALL SELECT 'process_rows' AS metric, COUNT(*) AS value FROM process "
                        "UNION ALL SELECT 'thread_rows' AS metric, COUNT(*) AS value FROM thread;"
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _run_trace_processor(self) -> None:
        output: dict[str, Any] = {
            "attempted": False,
            "available": False,
            "trace_exists": self.trace_path.exists(),
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error": None,
        }
        if not self.trace_path.exists():
            self._write_json(self.trace_processor_output_path, output)
            return
        try:
            setup = ensure_trace_processor(self.config.tools_dir)
            output["setup"] = setup
            output["available"] = bool(setup.get("available"))
            if not output["available"]:
                output["error"] = setup.get("error") or "trace_processor wrapper unavailable"
                self._write_json(self.trace_processor_output_path, output)
                return
            wrapper = setup["wrapper"]
            cmd = [
                self.config.python_executable,
                wrapper,
                "query",
                "-f",
                str(self.query_path),
                str(self.trace_path),
            ]
            output["attempted"] = True
            output["command"] = cmd
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            output["stdout"] = result.stdout
            output["stderr"] = result.stderr
            output["returncode"] = result.returncode
        except Exception as exc:
            output["error"] = str(exc)
            output["traceback"] = traceback.format_exc()
        self._write_json(self.trace_processor_output_path, output)

    def _adb_shell(self, command: str, *, timeout: int, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = [str(self._require_adb())]
        if self.config.device_serial:
            cmd += ["-s", self.config.device_serial]
        cmd += ["shell", command]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)

    def _adb_run(self, args: list[str], *, timeout: int, check: bool = True) -> subprocess.CompletedProcess[str]:
        cmd = [str(self._require_adb())]
        if self.config.device_serial:
            cmd += ["-s", self.config.device_serial]
        cmd += args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)

    def _require_adb(self) -> Path:
        if not self._adb_path:
            self._adb_path = discover_adb_path(self.config.adb_path)
        return self._adb_path

    def _record_error(self, exc: Exception) -> None:
        self.info.setdefault("errors", []).append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_perfetto_config(ftrace_events: list[str], atrace_categories: list[str], buffer_mb: int) -> str:
    lines = [
        f"buffers {{ size_kb: {max(1, int(buffer_mb)) * 1024} fill_policy: RING_BUFFER }}",
        "data_sources {",
        "  config {",
        '    name: "linux.ftrace"',
        "    ftrace_config {",
    ]
    for event in ftrace_events:
        lines.append(f'      ftrace_events: "{event}"')
    for category in atrace_categories:
        lines.append(f'      atrace_categories: "{category}"')
    lines.extend(
        [
            "    }",
            "  }",
            "}",
            "flush_period_ms: 5000",
            "",
        ]
    )
    return "\n".join(lines)


def parse_background_pid(stdout: str) -> str | None:
    for line in stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            return text
    matches = re.findall(r"\b\d+\b", stdout)
    return matches[-1] if matches else None


def parse_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("__WFGEN_") and line.endswith("__"):
            current = line.removeprefix("__WFGEN_").removesuffix("__")
            sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(value) for key, value in sections.items()}


def parse_key_value_kb(text: str) -> dict[str, float]:
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].rstrip(":")
            value = parse_float(parts[1])
            if value is not None:
                values[key] = value
    return values


def parse_equals(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().split("/")[-1]
        values[key] = value.strip()
    return values


def parse_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip().split()[0])
    except (ValueError, IndexError):
        return None


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "started_at_utc": None,
            "finished_at_utc": None,
            "duration_s": 0,
            "metrics": {},
        }
    metrics = {name: aggregate_metric(samples, name) for name in SUMMARY_METRICS}
    metrics = {key: value for key, value in metrics.items() if value["count"] > 0}
    started = min(float(sample["timestamp_epoch_s"]) for sample in samples)
    finished = max(float(sample["timestamp_epoch_s"]) for sample in samples)
    return {
        "sample_count": len(samples),
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat().replace("+00:00", "Z"),
        "duration_s": round(max(0.0, finished - started), 3),
        "metrics": metrics,
        "sample_errors": [sample.get("sample_error") for sample in samples if sample.get("sample_error")],
    }


def summarize_by_workflow(samples: list[dict[str, Any]], run_records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for record in run_records:
        started = record.get("started_epoch_s")
        finished = record.get("finished_epoch_s")
        if started is None or finished is None:
            workflow_samples = []
        else:
            workflow_samples = [
                sample
                for sample in samples
                if started <= float(sample.get("timestamp_epoch_s", 0.0)) <= finished
            ]
        payload = summarize_samples(workflow_samples)
        payload.update(
            {
                "workflow_index": record.get("workflow_index"),
                "workflow_id": record.get("workflow_id"),
                "name": record.get("name"),
                "status": record.get("status"),
                "result_json": record.get("result_json"),
                "started_epoch_s": started,
                "finished_epoch_s": finished,
            }
        )
        summaries[str(record.get("workflow_index"))] = payload
    return summaries


def aggregate_metric(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [parse_float(sample.get(key)) for sample in samples]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return {"count": 0, "avg": None, "min": None, "max": None, "last": None}
    return {
        "count": len(numeric),
        "avg": round(mean(numeric), 3),
        "min": round(min(numeric), 3),
        "max": round(max(numeric), 3),
        "last": round(numeric[-1], 3),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def shell_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


SAMPLE_SCRIPT = r"""
echo __WFGEN_PROC_STAT__
head -n 1 /proc/stat 2>/dev/null
echo __WFGEN_MEMINFO__
grep -E '^(MemTotal|MemAvailable|MemFree|Buffers|Cached|SwapTotal|SwapFree):' /proc/meminfo 2>/dev/null
echo __WFGEN_BATTERY__
for p in /sys/class/power_supply/battery/capacity /sys/class/power_supply/battery/temp /sys/class/power_supply/battery/current_now /sys/class/power_supply/battery/voltage_now; do
  if [ -r "$p" ]; then printf "%s=" "$(basename "$p")"; cat "$p" 2>/dev/null; fi
done
dumpsys battery 2>/dev/null | sed -n 's/^  level: /level=/p; s/^  temperature: /dumpsys_temp=/p; s/^  voltage: /dumpsys_voltage=/p; s/^  USB powered: /usb_powered=/p; s/^  AC powered: /ac_powered=/p'
echo __WFGEN_GPU__
for p in /sys/class/kgsl/kgsl-3d0/gpubusy /sys/class/kgsl/kgsl-3d0/gpuclk /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq /sys/class/kgsl/kgsl-3d0/devfreq/gpu_load; do
  if [ -r "$p" ]; then printf "%s=" "$(basename "$p")"; cat "$p" 2>/dev/null; fi
done
echo __WFGEN_THERMAL__
for d in /sys/class/thermal/thermal_zone*; do
  if [ -r "$d/temp" ]; then
    t="$(cat "$d/type" 2>/dev/null)"
    [ -n "$t" ] || t="$(basename "$d")"
    printf "%s=" "$t"
    cat "$d/temp" 2>/dev/null
  fi
done
echo __WFGEN_PROCESSES__
for n in llama-server llama-cli; do
  pid="$(pidof "$n" 2>/dev/null | cut -d ' ' -f 1)"
  if [ -n "$pid" ] && [ -r "/proc/$pid/status" ]; then
    rss="$(grep '^VmRSS:' "/proc/$pid/status" 2>/dev/null | tr -s ' ' | cut -d ' ' -f 2)"
    printf "name=%s pid=%s rss_kb=%s\n" "$n" "$pid" "$rss"
  fi
done
"""
