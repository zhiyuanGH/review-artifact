"""Runtime adapters for generated mobile workflow scripts.

The runtime intentionally mirrors the old pyagent assumptions:

* LLM nodes call an OpenAI-compatible llama-server style chat endpoint.
* App/tool/DNN nodes can be mapped to QNN `qnn-net-run` invocations.
* Dry-run mode is available so generated workflows can be debugged without an
  attached phone or deployed QNN bundles.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shlex
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from MOBench.tools.llm_tau import LlamaCliClient
except ImportError:
    from tools.llm_tau import LlamaCliClient


def load_runtime_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class MobileWorkflowRuntime:
    """Execute a generated workflow spec on a mobile-oriented runtime."""

    def __init__(self, config: dict[str, Any] | None = None, *, dry_run: bool = False):
        self.config = config or {}
        self.dry_run = dry_run

    def warmup_llama_cli(self, seen_keys: set[str] | None = None) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "skipped", "reason": "dry_run", "runs": []}
        seen = seen_keys if seen_keys is not None else set()
        runs = []
        for item in self._llama_cli_warmup_configs():
            key = self._llama_cli_warmup_key(item)
            if key in seen:
                runs.append({"status": "skipped", "reason": "already_warmed", "key": key, "label": item["label"]})
                continue
            seen.add(key)
            cfg = dict(item["config"])
            warmup_runs = int(cfg.get("warmup_runs") or 0)
            if warmup_runs <= 0:
                runs.append({"status": "skipped", "reason": "disabled", "key": key, "label": item["label"]})
                continue
            started = time.time()
            try:
                client = LlamaCliClient.from_config(cfg)
                result = client.warmup(runs=warmup_runs)
                result.update(
                    {
                        "key": key,
                        "label": item["label"],
                        "wall_ms": (time.time() - started) * 1000.0,
                    }
                )
                runs.append(result)
            except Exception as exc:
                runs.append(
                    {
                        "status": "error",
                        "key": key,
                        "label": item["label"],
                        "wall_ms": (time.time() - started) * 1000.0,
                        "error": str(exc),
                    }
                )
        status = "ok" if all(item.get("status") in {"ok", "skipped"} for item in runs) else "error"
        return {"status": status, "runs": runs}

    def execute_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        self._validate_workflow(workflow)
        waves = self._execution_waves(workflow)
        node_map = {node["node_id"]: node for node in workflow["nodes"]}
        context: dict[str, Any] = {"nodes": {}, "artifacts": {}}
        step_results: list[dict[str, Any]] = []
        start = time.time()

        for wave in waves:
            wave_start = time.time()
            wave_results = self._execute_wave(wave, node_map, workflow, context)
            step_results.extend(wave_results)
            for result in wave_results:
                context["nodes"][result["node_id"]] = result
                for name, value in result.get("artifacts", {}).items():
                    context["artifacts"][name] = value
            print(
                f"[wave {wave['wave_index']}] completed {len(wave_results)} nodes "
                f"in {time.time() - wave_start:.2f}s"
            )
            if any(not item.get("success", False) for item in wave_results):
                print(f"[wave {wave['wave_index']}] failed; stopping downstream execution")
                break

        return {
            "workflow_id": workflow.get("workflow_id"),
            "name": workflow.get("name"),
            "status": "completed" if self._completed_all_nodes(workflow, step_results) else "failed",
            "dry_run": self.dry_run,
            "started_at_utc": datetime.fromtimestamp(start, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_wall_ms": (time.time() - start) * 1000.0,
            "waves": waves,
            "steps": step_results,
            "artifacts": context["artifacts"],
        }

    def _execute_wave(
        self,
        wave: dict[str, Any],
        node_map: dict[str, dict[str, Any]],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        node_ids = wave["node_ids"]
        if len(node_ids) == 1:
            return [self.execute_node(node_map[node_ids[0]], workflow, context)]

        max_workers = min(
            len(node_ids),
            int(self.config.get("max_parallel_nodes", len(node_ids))),
        )
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_node_id = {
                executor.submit(self.execute_node, node_map[node_id], workflow, context): node_id
                for node_id in node_ids
            }
            for future in concurrent.futures.as_completed(future_to_node_id):
                results.append(future.result())
        return sorted(results, key=lambda item: node_ids.index(item["node_id"]))

    def execute_node(
        self,
        node: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        kind = node.get("kind", "app")
        if kind == "llm":
            return self._run_llm_node(node, workflow, context)
        if kind in {"app", "dnn"}:
            return self._run_tool_node(node, workflow, context)
        return self._run_control_or_data_node(node, workflow, context)

    def _run_llm_node(
        self,
        node: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        messages = self._build_llm_messages(node, workflow, context)
        if self.dry_run:
            return self._mock_node_result(
                node,
                {
                    "text": f"[dry-run llm output for {node['node_id']}]",
                    "messages": messages,
                },
            )

        llm_cfg = self.config.get("llm", {})
        runtime = str(llm_cfg.get("runtime") or "llama_server")
        if runtime == "mock":
            return self._mock_node_result(
                node,
                {
                    "text": f"[mock llm output for {node['node_id']}]",
                    "messages": messages,
                    "runtime": "mock",
                },
            )
        if runtime == "llama_cli":
            return self._run_llama_cli_node(node, messages, llm_cfg)
        if runtime == "ort_genai_cli":
            return self._run_ort_genai_cli_node(node, messages, llm_cfg)
        if runtime == "genie_cli":
            return self._run_genie_cli_node(node, messages, llm_cfg)
        if runtime != "llama_server":
            return self._node_result(
                node,
                success=False,
                wall_ms=0.0,
                raw={"error": f"Unsupported LLM node runtime: {runtime}"},
                artifacts={},
            )
        url = llm_cfg.get("url")
        if not url:
            raise ValueError("Missing runtime config key: llm.url")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Missing dependency: requests") from exc

        payload = {
            "messages": messages,
            "max_tokens": int(llm_cfg.get("max_tokens", 256)),
            "temperature": float(llm_cfg.get("temperature", 0.2)),
            "stream": False,
        }
        if llm_cfg.get("model"):
            payload["model"] = llm_cfg["model"]

        start = time.time()
        try:
            response = requests.post(url, json=payload, timeout=int(llm_cfg.get("timeout", 600)))
            response.raise_for_status()
            data = response.json()
            text = self._extract_chat_text(data)
            artifacts = self._output_artifacts(node, {"text": text})
            artifacts[f"{node['node_id']}_llm_metadata"] = {
                "model": payload.get("model") or data.get("model") or "",
                "runtime": "llama_server",
                "url": url,
            }
            return self._node_result(
                node,
                success=True,
                wall_ms=(time.time() - start) * 1000.0,
                raw={
                    "model": payload.get("model") or data.get("model") or "",
                    "url": url,
                    "response": data,
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            return self._node_result(
                node,
                success=False,
                wall_ms=(time.time() - start) * 1000.0,
                raw={"model": payload.get("model", ""), "url": url, "error": str(exc)},
                artifacts={},
            )

    def _run_llama_cli_node(
        self,
        node: dict[str, Any],
        messages: list[dict[str, str]],
        llm_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        start = time.time()
        try:
            cli_cfg = self._llama_cli_node_config(node, llm_cfg)
            client = LlamaCliClient.from_config(cli_cfg)
            response = client.chat(messages)
            text = response["text"]
            timings = response.get("timings") or {}
            artifacts = self._output_artifacts(node, {"text": text})
            artifacts[f"{node['node_id']}_llm_metadata"] = {
                "model": client.model_path,
                "runtime": "llama_cli",
                "target": client.target,
                "device": client.llama_device,
                "ngl": client.n_gpu_layers,
                "timings": timings,
            }
            return self._node_result(
                node,
                success=True,
                wall_ms=(time.time() - start) * 1000.0,
                raw={
                    "mode": "llama_cli",
                    "model": client.model_path,
                    "target": client.target,
                    "device": client.llama_device,
                    "ngl": client.n_gpu_layers,
                    "response_text": text,
                    "stdout": response.get("stdout", ""),
                    "stderr": response.get("stderr", ""),
                    "returncode": response.get("returncode"),
                    "command": response.get("command"),
                    "prompt_file": response.get("prompt_file"),
                    "timings": timings,
                    "inference_wall_ms": timings.get("inference_wall_ms"),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            return self._node_result(
                node,
                success=False,
                wall_ms=(time.time() - start) * 1000.0,
                raw={"mode": "llama_cli", "error": str(exc)},
                artifacts={},
            )

    def _run_ort_genai_cli_node(
        self,
        node: dict[str, Any],
        messages: list[dict[str, str]],
        llm_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        start = time.time()
        cfg = dict(llm_cfg.get("ort_genai") or {})
        prompt = self._messages_to_prompt(messages)
        original_prompt_chars = len(prompt)
        prompt = self._limit_prompt(prompt, int(cfg.get("prompt_char_limit") or 0))
        prompt_truncated = len(prompt) != original_prompt_chars
        max_tokens = int(cfg.get("max_tokens") or llm_cfg.get("max_tokens") or 64)
        timeout = int(cfg.get("timeout") or llm_cfg.get("timeout") or 240)
        runner_dir = self._require(cfg, "runner_dir")
        model_path = self._require(cfg, "model_path")
        jar = str(cfg.get("jar") or "oga-generate.jar")
        remote_cmd = (
            f"cd {shlex.quote(runner_dir)} && "
            "export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && "
            f"timeout {timeout} dalvikvm -Djava.library.path=. "
            f"-cp {shlex.quote(jar)} OgaGenerate "
            f"{shlex.quote(model_path)} {shlex.quote(prompt)} {int(max_tokens)}"
        )
        try:
            result = self._adb_shell(cfg, remote_cmd, timeout=timeout + 30)
            parse_error = ""
            try:
                data = self._parse_last_json_object(result.stdout)
            except Exception as exc:
                data = {}
                parse_error = str(exc)
            text = str(data.get("output") or "")
            success = result.returncode == 0 and data.get("status") == "ok"
            artifacts = self._output_artifacts(node, {"text": text}) if success else {}
            if success:
                artifacts[f"{node['node_id']}_llm_metadata"] = {
                    "runtime": "ort_genai_cli",
                    "model": data.get("model_path") or model_path,
                    "device": "none",
                    "model_load_ms": data.get("model_load_ms"),
                    "generation_ms": data.get("generation_ms"),
                    "prompt_tokens": data.get("prompt_tokens"),
                    "generated_tokens": data.get("generated_tokens"),
                    "prompt_truncated": prompt_truncated,
                    "prompt_chars": len(prompt),
                    "original_prompt_chars": original_prompt_chars,
                }
            raw = {
                "mode": "ort_genai_cli",
                "model": model_path,
                "device": "none",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": result.args,
                "response": data,
                "response_text": text,
                "inference_wall_ms": data.get("generation_ms"),
                "model_load_ms": data.get("model_load_ms"),
                "generation_ms": data.get("generation_ms"),
                "prompt_tokens": data.get("prompt_tokens"),
                "generated_tokens": data.get("generated_tokens"),
                "prompt_truncated": prompt_truncated,
                "prompt_chars": len(prompt),
                "original_prompt_chars": original_prompt_chars,
            }
            if parse_error:
                raw["error"] = parse_error
            return self._node_result(
                node,
                success=success,
                wall_ms=(time.time() - start) * 1000.0,
                raw=raw,
                artifacts=artifacts,
            )
        except Exception as exc:
            return self._node_result(
                node,
                success=False,
                wall_ms=(time.time() - start) * 1000.0,
                raw={"mode": "ort_genai_cli", "error": str(exc), "model": model_path},
                artifacts={},
            )

    def _run_genie_cli_node(
        self,
        node: dict[str, Any],
        messages: list[dict[str, str]],
        llm_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        start = time.time()
        cfg = dict(llm_cfg.get("genie") or {})
        bundle_dir = self._require(cfg, "bundle_dir")
        timeout = int(cfg.get("timeout") or llm_cfg.get("timeout") or 240)
        backend = str(cfg.get("backend") or "npu").lower()
        prompt = self._messages_to_prompt(messages)
        token = self._remote_token(node)
        prompt_remote = f"{bundle_dir.rstrip('/')}/wfgen_prompt_{token}.txt"
        profile_remote = f"{bundle_dir.rstrip('/')}/wfgen_profile_{token}.json"
        generated_config_remote = ""
        try:
            self._adb_write_text(cfg, prompt_remote, prompt, timeout=30)
            config_info = self._prepare_genie_config(cfg, node, backend, token, int(cfg.get("max_tokens") or llm_cfg.get("max_tokens") or 64))
            generated_config_remote = config_info.get("remote_config", "")
            remote_cmd = (
                f"cd {shlex.quote(bundle_dir)} && "
                f"rm -f {shlex.quote(profile_remote)} && "
                "export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && "
                "export ADSP_LIBRARY_PATH=.:$ADSP_LIBRARY_PATH && "
                f"timeout {timeout} ./genie-t2t-run "
                f"-c {shlex.quote(config_info['remote_config'])} "
                f"--prompt_file {shlex.quote(prompt_remote)} "
                f"--log {shlex.quote(str(cfg.get('log_level') or 'warn'))} "
                f"--profile {shlex.quote(profile_remote)}"
            )
            result = self._adb_shell(cfg, remote_cmd, timeout=timeout + 30)
            profile_text = self._adb_read_text(cfg, profile_remote, timeout=30, check=False)
            profile = self._loads_json(profile_text) if profile_text.strip() else {}
            profile_summary = self._parse_genie_profile(profile)
            text = self._extract_genie_text(result.stdout)
            success = result.returncode == 0
            artifacts = self._output_artifacts(node, {"text": text}) if success else {}
            if success:
                artifacts[f"{node['node_id']}_llm_metadata"] = {
                    "runtime": "genie_cli",
                    "bundle_dir": bundle_dir,
                    "config": config_info.get("base_remote_config"),
                    "generated_config": config_info.get("remote_config"),
                    "device": "HTP0" if backend == "npu" else backend,
                    "backend": backend,
                    "profile_summary": profile_summary,
                }
            return self._node_result(
                node,
                success=success,
                wall_ms=(time.time() - start) * 1000.0,
                raw={
                    "mode": "genie_cli",
                    "bundle_dir": bundle_dir,
                    "backend": backend,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": result.args,
                    "response_text": text,
                    "profile": profile,
                    "profile_summary": profile_summary,
                    "inference_wall_ms": self._profile_us_to_ms(profile_summary.get("query_us")),
                },
                artifacts=artifacts,
            )
        except Exception as exc:
            return self._node_result(
                node,
                success=False,
                wall_ms=(time.time() - start) * 1000.0,
                raw={"mode": "genie_cli", "error": str(exc), "bundle_dir": bundle_dir, "backend": backend},
                artifacts={},
            )
        finally:
            self._adb_rm(cfg, [prompt_remote, profile_remote, generated_config_remote], timeout=30)

    def _llama_cli_node_config(self, node: dict[str, Any], llm_cfg: dict[str, Any]) -> dict[str, Any]:
        cli_cfg = dict(llm_cfg.get("llama_cli") or {})
        cli_cfg.setdefault("max_tokens", llm_cfg.get("max_tokens", 256))
        cli_cfg.setdefault("temperature", llm_cfg.get("temperature", 0.2))
        cli_cfg.setdefault("timeout_seconds", llm_cfg.get("timeout", 600))
        overrides = llm_cfg.get("llama_cli_overrides") or {}
        override = overrides.get(node["node_id"]) or overrides.get(node.get("label", "")) or {}
        cli_cfg.update(override)
        return cli_cfg

    def _llama_cli_warmup_configs(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        llm_cfg = self.config.get("llm") or {}
        if llm_cfg.get("runtime") == "llama_cli":
            base = llm_cfg.get("llama_cli")
            if isinstance(base, dict):
                items.append({"label": "llm.default", "config": base})
            for node_id, cfg in (llm_cfg.get("llama_cli_overrides") or {}).items():
                if isinstance(cfg, dict):
                    items.append({"label": f"llm.{node_id}", "config": cfg})
        for entrypoint, cfg in (self.config.get("qnn_tools") or {}).items():
            if not isinstance(cfg, dict):
                continue
            if cfg.get("llm_runtime") == "llama_cli" or cfg.get("runtime") == "llama_cli":
                items.append({"label": f"tool.{entrypoint}", "config": cfg})
        return items

    @staticmethod
    def _llama_cli_warmup_key(item: dict[str, Any]) -> str:
        cfg = item.get("config") or {}
        keys = [
            "llama_cli_target",
            "target",
            "model_path",
            "llama_model_path",
            "model",
            "remote_dir",
            "binary",
            "llama_device",
            "device",
            "ngl",
            "hexagon_env",
            "ctx_size",
            "ubatch_size",
            "threads",
            "cpu_mask",
            "extra_args",
            "adb",
            "device_serial",
            "adb_serial",
        ]
        normalized = {key: cfg.get(key) for key in keys if key in cfg}
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True)

    def _run_tool_node(
        self,
        node: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        entrypoint = node.get("entrypoint") or node["node_id"]
        params = self._node_parameters(node)
        if self.dry_run:
            return self._mock_node_result(
                node,
                {
                    "entrypoint": entrypoint,
                    "parameters": params,
                    "upstream": self._upstream_results(node, workflow, context),
                },
            )

        tool_cfg = self._tool_config(entrypoint)
        mode = tool_cfg.get("mode", "qnn")
        if mode == "mock":
            return self._mock_node_result(node, {"entrypoint": entrypoint, "parameters": params})
        if mode == "json_cli":
            return self._run_json_cli_tool(node, tool_cfg, params, workflow, context)
        if mode == "shell":
            return self._run_shell_tool(node, tool_cfg, params)
        if mode == "qnn":
            return self._run_qnn_tool(node, tool_cfg, params)
        raise ValueError(f"Unsupported tool mode for {entrypoint}: {mode}")

    def _run_control_or_data_node(
        self,
        node: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return self._mock_node_result(
            node,
            {
                "parameters": self._node_parameters(node),
                "upstream": self._upstream_results(node, workflow, context),
            },
        )

    def _run_qnn_tool(
        self,
        node: dict[str, Any],
        tool_cfg: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        qnn_dir = self._require(tool_cfg, "qnn_dir")
        dlc_path = self._require(tool_cfg, "dlc_path")
        input_list = self._require(tool_cfg, "input_list")
        output_dir = self._require(tool_cfg, "output_dir")
        backend = tool_cfg.get("backend", "libQnnHtp.so")
        log_level = tool_cfg.get("log_level", "verbose")
        adsp_library_path = self._require(tool_cfg, "adsp_library_path")
        adb_path = self._require(tool_cfg, "adb")
        device_serial = tool_cfg.get("device")
        timeout = int(tool_cfg.get("timeout", 120))
        extra_args = tool_cfg.get("extra_args", [])

        ld = f"{qnn_dir}/lib:{qnn_dir}"
        extra = " " + " ".join(shlex.quote(str(arg)) for arg in extra_args) if extra_args else ""
        remote_cmd = (
            f"cd {shlex.quote(qnn_dir)} && "
            f"export LD_LIBRARY_PATH={shlex.quote(ld)}:$LD_LIBRARY_PATH && "
            f"export ADSP_LIBRARY_PATH={shlex.quote(adsp_library_path)} && "
            f"{shlex.quote(qnn_dir)}/qnn-net-run "
            f"--model libQnnModelDlc.so "
            f"--dlc_path {shlex.quote(dlc_path)} "
            f"--backend {shlex.quote(backend)} "
            f"--input_list {shlex.quote(input_list)} "
            f"--output_dir {shlex.quote(output_dir)} "
            f"--log_level {shlex.quote(log_level)}{extra}"
        )

        adb_cmd = [adb_path]
        if device_serial:
            adb_cmd += ["-s", device_serial]
        adb_cmd += ["shell", "sh", "-c", remote_cmd]

        start = time.time()
        try:
            result = subprocess.run(
                adb_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = result.returncode == 0
            return self._node_result(
                node,
                success=success,
                wall_ms=(time.time() - start) * 1000.0,
                raw={
                    "mode": "qnn",
                    "entrypoint": node.get("entrypoint"),
                    "parameters": params,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": adb_cmd,
                },
                artifacts=self._output_artifacts(
                    node,
                    {
                        "qnn_output_dir": output_dir,
                        "returncode": result.returncode,
                    },
                )
                if success
                else {},
            )
        except Exception as exc:
            return self._node_result(
                node,
                success=False,
                wall_ms=(time.time() - start) * 1000.0,
                raw={"mode": "qnn", "error": str(exc), "parameters": params},
                artifacts={},
            )

    def _run_json_cli_tool(
        self,
        node: dict[str, Any],
        tool_cfg: dict[str, Any],
        params: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        command = tool_cfg.get("command")
        if not command:
            raise ValueError(f"Missing json_cli command for tool {node.get('entrypoint')}")

        request = self._tool_request(node, tool_cfg, params, workflow, context)
        env = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        for key, value in (tool_cfg.get("env") or {}).items():
            env[str(key)] = str(value)

        shell = bool(tool_cfg.get("shell", False))
        if isinstance(command, list):
            run_command = [str(item) for item in command]
        elif shell:
            run_command = str(command)
        else:
            run_command = shlex.split(str(command))

        start = time.time()
        try:
            result = subprocess.run(
                run_command,
                input=json.dumps(request, ensure_ascii=True),
                capture_output=True,
                text=True,
                timeout=int(tool_cfg.get("timeout", 120)),
                cwd=tool_cfg.get("cwd") or project_root,
                env=env,
                shell=shell,
            )
            response = self._parse_json_cli_response(result.stdout)
            success = result.returncode == 0 and bool(response.get("success", False))
            artifacts = self._project_tool_artifacts(node, response.get("artifacts", {})) if success else {}
            raw = {
                "mode": "json_cli",
                "parameters": params,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": run_command,
                "response": response,
            }
            if not success and response.get("error"):
                raw["error"] = response["error"]
            return self._node_result(
                node,
                success=success,
                wall_ms=(time.time() - start) * 1000.0,
                raw=raw,
                artifacts=artifacts,
            )
        except Exception as exc:
            return self._node_result(
                node,
                success=False,
                wall_ms=(time.time() - start) * 1000.0,
                raw={"mode": "json_cli", "error": str(exc), "parameters": params},
                artifacts={},
            )

    def _run_shell_tool(
        self,
        node: dict[str, Any],
        tool_cfg: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        command = tool_cfg.get("command")
        if not command:
            raise ValueError(f"Missing shell command for tool {node.get('entrypoint')}")
        start = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=int(tool_cfg.get("timeout", 120)),
            )
            success = result.returncode == 0
            return self._node_result(
                node,
                success=success,
                wall_ms=(time.time() - start) * 1000.0,
                raw={
                    "mode": "shell",
                    "parameters": params,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": command,
                },
                artifacts=self._output_artifacts(node, {"stdout": result.stdout})
                if success
                else {},
            )
        except Exception as exc:
            return self._node_result(
                node,
                success=False,
                wall_ms=(time.time() - start) * 1000.0,
                raw={"mode": "shell", "error": str(exc), "parameters": params},
                artifacts={},
            )

    def _tool_config(self, entrypoint: str) -> dict[str, Any]:
        default_cfg = dict(self.config.get("qnn_default", {}))
        tools = self.config.get("qnn_tools", {})
        specific = tools.get(entrypoint, {})
        default_cfg.update(specific)
        return default_cfg

    def _tool_request(
        self,
        node: dict[str, Any],
        tool_cfg: dict[str, Any],
        params: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_only = {"command", "mode", "timeout", "shell", "env", "cwd"}
        return {
            "entrypoint": node.get("entrypoint") or node["node_id"],
            "node_id": node["node_id"],
            "parameters": params,
            "upstream": self._upstream_results(node, workflow, context),
            "tool_config": {key: value for key, value in tool_cfg.items() if key not in runtime_only},
            "workflow": {
                "workflow_id": workflow.get("workflow_id"),
                "name": workflow.get("name"),
            },
        }

    def _build_llm_prompt(
        self,
        node: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        upstream = self._upstream_results(node, workflow, context)
        return json.dumps(
            {
                "workflow": workflow.get("name"),
                "node": {
                    "node_id": node.get("node_id"),
                    "purpose": node.get("purpose"),
                    "parameters": self._node_parameters(node),
                },
                "upstream_results": upstream,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _build_llm_messages(
        self,
        node: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        params = self._node_parameters(node)
        system_prompt = str(
            params.get("system_prompt")
            or "You are an LLM reasoning node in a generated workflow. Return concise JSON when possible. Do not invent IDs."
        )
        user_instruction = str(params.get("instruction") or "").strip()
        visible_params = {
            key: value
            for key, value in params.items()
            if key not in {"system_prompt", "instruction"}
        }
        prompt_payload = {
            "workflow": workflow.get("name"),
            "node": {
                "node_id": node.get("node_id"),
                "purpose": node.get("purpose"),
                "entrypoint": node.get("entrypoint"),
                "parameters": visible_params,
                "declared_outputs": node.get("outputs") or [],
            },
            "upstream_results": self._upstream_results(node, workflow, context),
        }
        parts = []
        if user_instruction:
            parts.append(user_instruction)
        parts.extend(
            [
                "Use the upstream results below as the only runtime evidence.",
                "Return the node output only, preferably as compact JSON.",
                json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            ]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"{str(message.get('role') or 'user').upper()}:\n{message.get('content') or ''}"
            for message in messages
        )

    @staticmethod
    def _limit_prompt(prompt: str, char_limit: int) -> str:
        if char_limit <= 0 or len(prompt) <= char_limit:
            return prompt
        marker = "\n\n...[truncated for runtime context limit]...\n\n"
        if char_limit <= len(marker) + 64:
            return prompt[:char_limit]
        prefix_len = max(64, char_limit // 3)
        suffix_len = max(64, char_limit - prefix_len - len(marker))
        return prompt[:prefix_len] + marker + prompt[-suffix_len:]

    def _output_artifacts(self, node: dict[str, Any], value: Any) -> dict[str, Any]:
        outputs = node.get("outputs") or []
        if not outputs:
            return {node["node_id"]: value}
        return {artifact["name"]: value for artifact in outputs}

    def _project_tool_artifacts(self, node: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
        projected = dict(artifacts or {})
        outputs = node.get("outputs") or []
        if not outputs:
            return projected
        for output in outputs:
            name = output["name"]
            if name in projected:
                continue
            mapped = self._artifact_alias_value(name, projected)
            if mapped is not None:
                projected[name] = mapped
            elif len(projected) == 1:
                projected[name] = next(iter(projected.values()))
        return projected

    @staticmethod
    def _artifact_alias_value(name: str, artifacts: dict[str, Any]) -> Any:
        aliases = {
            "parsed_slots": ["tau_slots"],
            "matched_variants": ["retail_variant_matches"],
            "ranked_options": ["ranked_flight_options"],
            "payment_resolution": ["payment_resolution"],
            "confirmation": ["confirmation_decision"],
            "order_details": ["order"],
            "order_record": ["order"],
            "reservation_details": ["reservation", "order"],
            "user_identity": ["user", "user_id"],
            "user_record": ["user"],
        }
        if name.endswith("_product") and "product" in artifacts:
            return artifacts["product"]
        if name.endswith("_catalog_entry") and "product" in artifacts:
            return artifacts["product"]
        if name.endswith("_details") and name.startswith("order") and "order" in artifacts:
            return artifacts["order"]
        if name.startswith("resolved_") and "retail_variant_matches" in artifacts:
            return artifacts["retail_variant_matches"]
        for alias in aliases.get(name, []):
            if alias in artifacts:
                return artifacts[alias]
        return None

    def _mock_node_result(self, node: dict[str, Any], value: Any) -> dict[str, Any]:
        return self._node_result(
            node,
            success=True,
            wall_ms=0.0,
            raw={"dry_run": True, "value": value},
            artifacts=self._output_artifacts(node, value),
        )

    def _node_result(
        self,
        node: dict[str, Any],
        *,
        success: bool,
        wall_ms: float,
        raw: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "node_id": node["node_id"],
            "kind": node.get("kind"),
            "entrypoint": node.get("entrypoint"),
            "success": success,
            "wall_ms": wall_ms,
            "raw": raw,
            "artifacts": artifacts,
        }

    def _upstream_results(
        self,
        node: dict[str, Any],
        workflow: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        upstream: dict[str, Any] = {}
        node_id = node["node_id"]
        for edge in workflow.get("edges", []):
            if edge["to_node"] != node_id:
                continue
            artifact = edge.get("artifact", {})
            name = artifact.get("name") or edge["from_node"]
            producer = context["nodes"].get(edge["from_node"], {})
            producer_artifacts = producer.get("artifacts", {})
            upstream[name] = {
                "from_node": edge["from_node"],
                "success": producer.get("success"),
                "value": producer_artifacts.get(name, context["artifacts"].get(name)),
            }
        return upstream

    @staticmethod
    def _completed_all_nodes(workflow: dict[str, Any], step_results: list[dict[str, Any]]) -> bool:
        if not step_results:
            return False
        expected = {node["node_id"] for node in workflow["nodes"]}
        completed = {item["node_id"] for item in step_results if item.get("success", False)}
        return expected == completed

    @staticmethod
    def _node_parameters(node: dict[str, Any]) -> dict[str, Any]:
        return {
            item["name"]: MobileWorkflowRuntime._parse_parameter_value(item.get("value", ""))
            for item in node.get("parameters", [])
        }

    @staticmethod
    def _parse_parameter_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def _extract_chat_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    @staticmethod
    def _parse_json_cli_response(stdout: str) -> dict[str, Any]:
        text = stdout.strip()
        if not text:
            return {"success": False, "artifacts": {}, "error": "json_cli produced no stdout"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.rfind("\n{")
            if start >= 0:
                return json.loads(text[start + 1 :])
            raise

    @staticmethod
    def _parse_last_json_object(stdout: str) -> dict[str, Any]:
        text = stdout.strip()
        if not text:
            raise ValueError("Command produced no stdout JSON.")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.rfind("\n{")
            if start >= 0:
                return json.loads(text[start + 1 :])
            start = text.find("{")
            if start >= 0:
                return json.loads(text[start:])
            raise

    @staticmethod
    def _loads_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            return {}
        return json.loads(stripped)

    def _prepare_genie_config(
        self,
        cfg: dict[str, Any],
        node: dict[str, Any],
        backend: str,
        token: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        bundle_dir = self._require(cfg, "bundle_dir")
        base_config = str(cfg.get("config") or "auto")
        if base_config == "auto":
            base_config = self._auto_genie_config(cfg, bundle_dir)
        base_remote = base_config if base_config.startswith("/") else f"{bundle_dir.rstrip('/')}/{base_config}"
        config_text = self._adb_read_text(cfg, base_remote, timeout=30)
        config = self._loads_json(config_text)
        self._mutate_genie_config(config, backend, max_tokens)
        remote_config = f"{bundle_dir.rstrip('/')}/wfgen_config_{self._safe_remote_part(node['node_id'])}_{token}.json"
        self._adb_write_text(cfg, remote_config, json.dumps(config, ensure_ascii=False), timeout=30)
        return {
            "base_remote_config": base_remote,
            "remote_config": remote_config,
            "backend": backend,
        }

    def _auto_genie_config(self, cfg: dict[str, Any], bundle_dir: str) -> str:
        proc = self._adb_shell(cfg, "getprop ro.soc.model", timeout=15, check=False)
        soc = (proc.stdout or "").strip()
        if soc == "SM8750":
            return "genie_config_sm8750.json"
        return "genie_config.json"

    @staticmethod
    def _mutate_genie_config(config: dict[str, Any], backend: str, max_tokens: int) -> None:
        dialog = config.setdefault("dialog", {})
        if max_tokens > 0:
            dialog["max-num-tokens"] = int(max_tokens)
        engine = dialog.setdefault("engine", {})
        current_backend = engine.setdefault("backend", {})
        if backend == "npu":
            htp = current_backend.get("QnnHtp") if isinstance(current_backend.get("QnnHtp"), dict) else {}
            engine["backend"] = {
                "version": int(current_backend.get("version", 1)),
                "type": "QnnHtp",
                "QnnHtp": htp or {
                    "version": 1,
                    "use-mmap": True,
                    "spill-fill-bufsize": 0,
                    "mmap-budget": 0,
                    "poll": True,
                    "cpu-mask": "0xe0",
                    "allow-async-init": False,
                },
            }
            if current_backend.get("extensions"):
                engine["backend"]["extensions"] = current_backend["extensions"]
            return
        if backend == "gpu":
            engine["backend"] = {"version": int(current_backend.get("version", 1)), "type": "QnnGpu"}
            return
        engine["backend"] = {
            "version": int(current_backend.get("version", 1)),
            "type": "QnnGenAiTransformer",
            "QnnGenAiTransformer": {"version": 1, "kv-quantization": False},
        }

    @staticmethod
    def _parse_genie_profile(profile: dict[str, Any]) -> dict[str, Any]:
        events = [
            event
            for component in profile.get("components", [])
            for event in component.get("events", [])
            if isinstance(event, dict)
        ]
        create = next((event for event in events if event.get("type") == "GenieDialog_create"), {})
        query = next((event for event in events if event.get("type") == "GenieDialog_query"), {})
        return {
            "init_us": MobileWorkflowRuntime._metric_value(create, "init-time"),
            "query_us": query.get("duration"),
            "prompt_tokens": MobileWorkflowRuntime._metric_value(query, "num-prompt-tokens"),
            "prompt_processing_rate": MobileWorkflowRuntime._metric_value(query, "prompt-processing-rate"),
            "time_to_first_token_us": MobileWorkflowRuntime._metric_value(query, "time-to-first-token"),
            "generated_tokens": MobileWorkflowRuntime._metric_value(query, "num-generated-tokens"),
            "token_generation_rate": MobileWorkflowRuntime._metric_value(query, "token-generation-rate"),
            "token_generation_time_us": MobileWorkflowRuntime._metric_value(query, "token-generation-time"),
        }

    @staticmethod
    def _metric_value(event: dict[str, Any], key: str, fallback: Any = None) -> Any:
        value = event.get(key)
        if isinstance(value, dict):
            return value.get("value")
        return fallback if value is None else value

    @staticmethod
    def _extract_genie_text(stdout: str) -> str:
        text = stdout or ""
        begin = text.find("[BEGIN]:")
        end = text.find("[END]", begin)
        if begin >= 0 and end >= 0:
            return text[begin + len("[BEGIN]:") : end].strip()
        return text.strip()

    @staticmethod
    def _profile_us_to_ms(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value) / 1000.0
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _remote_token(node: dict[str, Any]) -> str:
        return f"{os.getpid()}_{int(time.time() * 1000000)}_{MobileWorkflowRuntime._safe_remote_part(str(node.get('node_id') or 'node'))}"

    @staticmethod
    def _safe_remote_part(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:64] or "item"

    def _adb_command(self, cfg: dict[str, Any]) -> list[str]:
        adb = str(cfg.get("adb") or "adb")
        command = [adb]
        serial = cfg.get("device_serial") or cfg.get("adb_serial")
        if serial:
            command.extend(["-s", str(serial)])
        return command

    def _adb_shell(
        self,
        cfg: dict[str, Any],
        remote_cmd: str,
        *,
        timeout: int,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = self._adb_command(cfg) + ["shell", remote_cmd]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or f"adb shell failed with code {result.returncode}")
        return result

    def _adb_write_text(self, cfg: dict[str, Any], remote_path: str, text: str, *, timeout: int) -> None:
        command = self._adb_command(cfg) + ["shell", f"cat > {shlex.quote(remote_path)}"]
        result = subprocess.run(command, input=text, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or f"Failed to write remote file: {remote_path}")

    def _adb_read_text(
        self,
        cfg: dict[str, Any],
        remote_path: str,
        *,
        timeout: int,
        check: bool = True,
    ) -> str:
        result = self._adb_shell(cfg, f"cat {shlex.quote(remote_path)}", timeout=timeout, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or f"Failed to read remote file: {remote_path}")
        return result.stdout or ""

    def _adb_rm(self, cfg: dict[str, Any], remote_paths: list[str], *, timeout: int) -> None:
        paths = [path for path in remote_paths if path]
        if not paths:
            return
        quoted = " ".join(shlex.quote(path) for path in paths)
        self._adb_shell(cfg, f"rm -f {quoted}", timeout=timeout, check=False)

    @staticmethod
    def _require(cfg: dict[str, Any], key: str) -> Any:
        value = cfg.get(key)
        if value is None or value == "":
            raise KeyError(f"Missing runtime config key: {key}")
        return value

    def _execution_waves(self, workflow: dict[str, Any]) -> list[dict[str, Any]]:
        hints = workflow.get("execution_hints", {})
        waves = hints.get("parallel_waves") or []
        if waves:
            return [
                {
                    "wave_index": int(wave["wave_index"]),
                    "node_ids": list(wave["node_ids"]),
                    "reason": wave.get("reason", ""),
                }
                for wave in sorted(waves, key=lambda item: item["wave_index"])
            ]
        return self._topological_waves(workflow)

    @staticmethod
    def _topological_waves(workflow: dict[str, Any]) -> list[dict[str, Any]]:
        node_ids = [node["node_id"] for node in workflow["nodes"]]
        outgoing = {node_id: [] for node_id in node_ids}
        indegree = {node_id: 0 for node_id in node_ids}
        for edge in workflow.get("edges", []):
            outgoing[edge["from_node"]].append(edge["to_node"])
            indegree[edge["to_node"]] += 1

        ready = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
        waves: list[dict[str, Any]] = []
        visited = 0
        while ready:
            wave_ids = list(ready)
            ready.clear()
            waves.append(
                {
                    "wave_index": len(waves),
                    "node_ids": wave_ids,
                    "reason": "Derived by Kahn topological layering.",
                }
            )
            for node_id in wave_ids:
                visited += 1
                for dst in outgoing[node_id]:
                    indegree[dst] -= 1
                    if indegree[dst] == 0:
                        ready.append(dst)
        if visited != len(node_ids):
            raise ValueError(f"Workflow has a cycle: {workflow.get('workflow_id')}")
        return waves

    @staticmethod
    def _validate_workflow(workflow: dict[str, Any]) -> None:
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("Workflow must contain non-empty nodes")
        node_ids = [node.get("node_id") for node in nodes]
        if any(not node_id for node_id in node_ids):
            raise ValueError("Every node must contain node_id")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Duplicate node_id in workflow")
        node_set = set(node_ids)
        for edge in workflow.get("edges", []):
            if edge.get("from_node") not in node_set:
                raise ValueError(f"Unknown edge source: {edge.get('from_node')}")
            if edge.get("to_node") not in node_set:
                raise ValueError(f"Unknown edge target: {edge.get('to_node')}")
        waves = workflow.get("execution_hints", {}).get("parallel_waves") or []
        if not waves:
            return
        seen: set[str] = set()
        wave_position: dict[str, int] = {}
        for wave in waves:
            for node_id in wave.get("node_ids", []):
                if node_id not in node_set:
                    raise ValueError(f"Unknown node in execution wave: {node_id}")
                if node_id in seen:
                    raise ValueError(f"Duplicate node in execution waves: {node_id}")
                seen.add(node_id)
                wave_position[node_id] = int(wave["wave_index"])
        missing = node_set - seen
        if missing:
            raise ValueError(f"Execution waves omit nodes: {sorted(missing)}")
        for edge in workflow.get("edges", []):
            if wave_position[edge["from_node"]] >= wave_position[edge["to_node"]]:
                raise ValueError(
                    "Execution waves violate dependency order: "
                    f"{edge['from_node']} -> {edge['to_node']}"
                )
