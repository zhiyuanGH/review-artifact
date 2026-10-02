"""Tau-bench inference tools backed by local llama inference.

The tools in this module are logical inference tools. They can run through a
shared llama-server endpoint or as direct llama-cli invocations on the phone.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .abi import fail, ok


LLM_TAU_TOOLS = {
    "extract_tau_slots",
    "match_retail_product_variant",
    "rank_airline_flight_options",
    "detect_confirmation_or_revision",
    "resolve_payment_preference",
}

DEFAULT_DATASET_PATH = Path("data/tau_bench_inference_subset_12.json")

ARTIFACT_NAMES = {
    "extract_tau_slots": "tau_slots",
    "match_retail_product_variant": "retail_variant_matches",
    "rank_airline_flight_options": "ranked_flight_options",
    "detect_confirmation_or_revision": "confirmation_decision",
    "resolve_payment_preference": "payment_resolution",
}


class LlamaServerError(RuntimeError):
    pass


class LlamaCliError(RuntimeError):
    pass


def run_llm_tau_tool(entrypoint: str, request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    if entrypoint not in LLM_TAU_TOOLS:
        return fail(f"Unsupported llama tau tool: {entrypoint}", started_at=started_at)

    params = request.get("parameters") or {}
    tool_cfg = request.get("tool_config") or {}
    sample = resolve_sample(params, tool_cfg)
    prompt_payload = build_prompt_payload(entrypoint, request, params, sample, tool_cfg)
    debug_raw: dict[str, Any] = {}

    try:
        if truthy(tool_cfg.get("mock_llm") or os.environ.get("WFGEN_LLM_TOOL_MOCK")):
            parsed = mock_tool_output(entrypoint, prompt_payload)
            raw = {
                "mode": "mock_llm",
                "entrypoint": entrypoint,
                "model": tool_cfg.get("model", ""),
                "url": tool_cfg.get("url") or tool_cfg.get("llm_url") or "",
                "backend": tool_cfg.get("backend") or tool_cfg.get("llama_backend") or "",
                "mg": tool_cfg.get("mg", ""),
                "prompt_payload": prompt_payload,
            }
        else:
            runtime = str(tool_cfg.get("llm_runtime") or tool_cfg.get("runtime") or "llama_server")
            messages = build_messages(entrypoint, prompt_payload)
            if runtime == "llama_cli":
                client = LlamaCliClient.from_config(tool_cfg)
                response = client.chat(messages)
                text = response["text"]
                debug_raw = {
                    "mode": "llama_cli",
                    "entrypoint": entrypoint,
                    "model": client.model_path,
                    "target": client.target,
                    "response_text": text,
                    "returncode": response.get("returncode"),
                    "stderr": response.get("stderr", "")[-4000:],
                    "command": response.get("command"),
                    "timings": response.get("timings") or {},
                }
                parsed = parse_json_object(text)
                raw = {
                    "mode": "llama_cli",
                    "entrypoint": entrypoint,
                    "model": client.model_path,
                    "target": client.target,
                    "backend": tool_cfg.get("backend") or tool_cfg.get("llama_backend"),
                    "mg": tool_cfg.get("mg"),
                    "prompt_payload": prompt_payload,
                    "response_text": text,
                    "stdout": response.get("stdout", ""),
                    "stderr": response.get("stderr", ""),
                    "returncode": response.get("returncode"),
                    "command": response.get("command"),
                    "prompt_file": response.get("prompt_file"),
                    "timings": response.get("timings") or {},
                    "inference_wall_ms": (response.get("timings") or {}).get("inference_wall_ms"),
                }
            elif runtime == "llama_server":
                client = LlamaServerClient.from_config(tool_cfg)
                response = client.chat(messages)
                text = extract_chat_text(response)
                debug_raw = {
                    "mode": "llama_server",
                    "entrypoint": entrypoint,
                    "model": client.model,
                    "url": client.url,
                    "backend": tool_cfg.get("backend") or tool_cfg.get("llama_backend"),
                    "mg": tool_cfg.get("mg"),
                    "response_text": text,
                    "response": compact_response(response),
                }
                parsed = parse_json_object(text)
                raw = {
                    "mode": "llama_server",
                    "entrypoint": entrypoint,
                    "model": client.model,
                    "url": client.url,
                    "backend": tool_cfg.get("backend") or tool_cfg.get("llama_backend"),
                    "mg": tool_cfg.get("mg"),
                    "prompt_payload": prompt_payload,
                    "response_text": text,
                    "response": compact_response(response),
                }
            else:
                raise ValueError(f"Unsupported LLM tool runtime: {runtime}")

        normalized = apply_obvious_fallbacks(entrypoint, normalize_tool_output(entrypoint, parsed), prompt_payload)
        artifacts = {
            ARTIFACT_NAMES[entrypoint]: normalized,
            "llm_tool_metadata": {
                "entrypoint": entrypoint,
                "model": raw.get("model", tool_cfg.get("model", "")),
                "runtime": raw["mode"],
                "backend": raw.get("backend") or tool_cfg.get("backend") or tool_cfg.get("llama_backend") or "",
                "mg": raw.get("mg", tool_cfg.get("mg", "")),
            },
        }
        return ok(artifacts=artifacts, raw=raw, started_at=started_at)
    except Exception as exc:
        return fail(
            str(exc),
            raw={
                "entrypoint": entrypoint,
                "parameters": params,
                "sample_id": sample.get("experiment_id") or sample.get("task_id"),
                "debug": debug_raw,
            },
            started_at=started_at,
        )


class LlamaServerClient:
    def __init__(
        self,
        *,
        url: str,
        model: str = "",
        max_tokens: int = 96,
        temperature: float = 0.0,
        timeout: int = 120,
        json_mode: bool = True,
    ) -> None:
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LlamaServerClient":
        url = (
            cfg.get("url")
            or cfg.get("llm_url")
            or os.environ.get("WFGEN_LLAMA_URL")
            or "http://localhost:8015/v1/chat/completions"
        )
        return cls(
            url=str(url),
            model=str(cfg.get("model") or os.environ.get("WFGEN_LLAMA_MODEL") or ""),
            max_tokens=int(cfg.get("max_tokens", 96)),
            temperature=float(cfg.get("temperature", 0.0)),
            timeout=int(cfg.get("timeout", 120)),
            json_mode=not truthy(cfg.get("disable_json_mode")),
        )

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            return self._post_json(payload)
        except LlamaServerError as exc:
            # Some older llama-server builds have incomplete OpenAI JSON-mode
            # compatibility. The prompt still strongly constrains JSON output.
            if "response_format" in payload:
                payload.pop("response_format", None)
                return self._post_json(payload)
            raise exc

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"llama-server HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LlamaServerError(f"llama-server request failed: {exc}") from exc


class LlamaCliClient:
    def __init__(
        self,
        *,
        target: str,
        model_path: str,
        max_tokens: int = 96,
        temperature: float = 0.0,
        timeout: int = 240,
        adb: str = "adb",
        device_serial: str = "",
        remote_dir: str = "/data/local/tmp/llama.cpp",
        remote_prompt_dir: str = "/data/local/tmp/wfgen_llama_prompts",
        binary: str = "llama-cli",
        llama_device: str = "HTP0",
        n_gpu_layers: int = 99,
        hexagon_env: bool = True,
        ctx_size: int = 8192,
        ubatch_size: int = 1024,
        threads: int = 6,
        cpu_mask: str = "0xfc",
        no_mmap: bool = True,
        json_schema: str = "{}",
        extra_args: list[str] | None = None,
        warmup_runs: int = 0,
        warmup_max_tokens: int = 1,
        warmup_prompt: str = "OK",
    ) -> None:
        self.target = target
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.adb = adb
        self.device_serial = device_serial
        self.remote_dir = remote_dir
        self.remote_prompt_dir = remote_prompt_dir
        self.binary = binary
        self.llama_device = llama_device
        self.n_gpu_layers = n_gpu_layers
        self.hexagon_env = hexagon_env
        self.ctx_size = ctx_size
        self.ubatch_size = ubatch_size
        self.threads = threads
        self.cpu_mask = cpu_mask
        self.no_mmap = no_mmap
        self.json_schema = json_schema
        self.extra_args = extra_args or []
        self.warmup_runs = max(0, warmup_runs)
        self.warmup_max_tokens = max(1, warmup_max_tokens)
        self.warmup_prompt = warmup_prompt

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LlamaCliClient":
        llama_device = str(cfg.get("llama_device") or cfg.get("device") or "HTP0")
        n_gpu_layers = int(cfg.get("ngl", 0 if llama_device == "none" else 99))
        return cls(
            target=str(cfg.get("llama_cli_target") or cfg.get("target") or "phone"),
            model_path=str(cfg.get("model_path") or cfg.get("llama_model_path") or cfg.get("model") or ""),
            max_tokens=int(cfg.get("max_tokens", 96)),
            temperature=float(cfg.get("temperature", 0.0)),
            timeout=int(cfg.get("llama_timeout") or cfg.get("timeout_seconds") or cfg.get("timeout", 240)),
            adb=str(cfg.get("adb") or os.environ.get("ADB") or "adb"),
            device_serial=str(cfg.get("device_serial") or cfg.get("adb_serial") or ""),
            remote_dir=str(cfg.get("remote_dir") or "/data/local/tmp/llama.cpp"),
            remote_prompt_dir=str(cfg.get("remote_prompt_dir") or "/data/local/tmp/wfgen_llama_prompts"),
            binary=str(cfg.get("binary") or "llama-cli"),
            llama_device=llama_device,
            n_gpu_layers=n_gpu_layers,
            hexagon_env=truthy(cfg.get("hexagon_env")) if "hexagon_env" in cfg else llama_device.startswith("HTP"),
            ctx_size=int(cfg.get("ctx_size", 8192)),
            ubatch_size=int(cfg.get("ubatch_size", 1024)),
            threads=int(cfg.get("threads", 6)),
            cpu_mask=str(cfg.get("cpu_mask", "0xfc")),
            no_mmap=not truthy(cfg.get("mmap")),
            json_schema=str(cfg.get("json_schema", "{}")),
            extra_args=[str(item) for item in cfg.get("extra_args", [])],
            warmup_runs=int(cfg.get("warmup_runs", 0)),
            warmup_max_tokens=int(cfg.get("warmup_max_tokens", 1)),
            warmup_prompt=str(cfg.get("warmup_prompt", "OK")),
        )

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self.model_path:
            raise LlamaCliError("llama_cli requires model_path or llama_model_path")
        prompt = self._render_prompt(messages)
        if self.target == "phone":
            return self._phone_complete(prompt, messages[0]["content"])
        if self.target == "local":
            return self._local_complete(prompt, messages[0]["content"])
        raise LlamaCliError(f"Unsupported llama_cli target: {self.target}")

    def warmup(self, *, runs: int | None = None) -> dict[str, Any]:
        run_count = self.warmup_runs if runs is None else max(0, runs)
        if run_count <= 0:
            return {"status": "skipped", "runs": [], "run_count": 0, "total_wall_ms": 0.0}
        system_prompt = "You are a warmup request. Reply with OK only."
        warmup_runs = []
        started = time.time()
        for index in range(run_count):
            run_started = time.time()
            try:
                if self.target == "phone":
                    response = self._phone_complete(
                        self.warmup_prompt,
                        system_prompt,
                        max_tokens_override=self.warmup_max_tokens,
                    )
                elif self.target == "local":
                    response = self._local_complete(
                        self.warmup_prompt,
                        system_prompt,
                        max_tokens_override=self.warmup_max_tokens,
                    )
                else:
                    raise LlamaCliError(f"Unsupported llama_cli target: {self.target}")
                warmup_runs.append(
                    {
                        "index": index,
                        "status": "ok",
                        "wall_ms": (time.time() - run_started) * 1000.0,
                        "returncode": response.get("returncode"),
                        "command": response.get("command"),
                        "prompt_file": response.get("prompt_file"),
                        "timings": response.get("timings") or {},
                        "stderr_tail": str(response.get("stderr") or "")[-2000:],
                    }
                )
            except Exception as exc:
                warmup_runs.append(
                    {
                        "index": index,
                        "status": "error",
                        "wall_ms": (time.time() - run_started) * 1000.0,
                        "error": str(exc),
                    }
                )
        status = "ok" if all(item.get("status") == "ok" for item in warmup_runs) else "error"
        return {
            "status": status,
            "runs": warmup_runs,
            "run_count": run_count,
            "total_wall_ms": (time.time() - started) * 1000.0,
            "model": self.model_path,
            "target": self.target,
            "device": self.llama_device,
            "ngl": self.n_gpu_layers,
        }

    def _phone_complete(
        self,
        prompt: str,
        system_prompt: str,
        *,
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        token = f"{os.getpid()}_{int(time.time() * 1000)}"
        prompt_file = f"{self.remote_prompt_dir}/prompt_{token}.txt"
        adb = [self.adb]
        if self.device_serial:
            adb += ["-s", self.device_serial]
        self._run(adb + ["shell", "mkdir", "-p", self.remote_prompt_dir], timeout=30)
        self._run(
            adb + ["shell", "sh", "-c", f"cat > {shlex.quote(prompt_file)}"],
            input_text=prompt,
            timeout=60,
        )
        remote_cmd = self._remote_command(prompt_file, system_prompt, max_tokens_override=max_tokens_override)
        try:
            result = self._run(adb + ["shell", "sh", "-c", remote_cmd], timeout=self.timeout)
            timings = parse_llama_perf_timings(result.stdout + "\n" + result.stderr)
            return {
                "text": self._model_text(result.stdout),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": [*adb, "shell", "sh", "-c", remote_cmd],
                "prompt_file": prompt_file,
                "timings": timings,
            }
        finally:
            self._run(adb + ["shell", "rm", "-f", prompt_file], timeout=30, check=False)

    def _local_complete(
        self,
        prompt: str,
        system_prompt: str,
        *,
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        import tempfile

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write(prompt)
            prompt_file = handle.name
        try:
            command = self._local_command(prompt_file, system_prompt, max_tokens_override=max_tokens_override)
            result = self._run(command, timeout=self.timeout)
            timings = parse_llama_perf_timings(result.stdout + "\n" + result.stderr)
            return {
                "text": self._model_text(result.stdout),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": command,
                "prompt_file": prompt_file,
                "timings": timings,
            }
        finally:
            Path(prompt_file).unlink(missing_ok=True)

    def _remote_command(
        self,
        prompt_file: str,
        system_prompt: str,
        *,
        max_tokens_override: int | None = None,
    ) -> str:
        lib_dir = f"{self.remote_dir}/lib"
        binary = f"{self.remote_dir}/bin/{self.binary}"
        parts = [
            "cd",
            shlex.quote(self.remote_dir),
            "&&",
            f"LD_LIBRARY_PATH={shlex.quote(lib_dir)}",
            f"ADSP_LIBRARY_PATH={shlex.quote(lib_dir)}",
            f"GGML_HEXAGON_NDEV={1 if self.hexagon_env else 0}",
            shlex.quote(binary),
            *self._common_cli_args(prompt_file, system_prompt, max_tokens_override=max_tokens_override),
        ]
        return " ".join(parts)

    def _local_command(
        self,
        prompt_file: str,
        system_prompt: str,
        *,
        max_tokens_override: int | None = None,
    ) -> list[str]:
        binary = self.binary
        return [binary, *self._common_cli_args(prompt_file, system_prompt, quote=False, max_tokens_override=max_tokens_override)]

    def _common_cli_args(
        self,
        prompt_file: str,
        system_prompt: str,
        *,
        quote: bool = True,
        max_tokens_override: int | None = None,
    ) -> list[str]:
        def value(text: object) -> str:
            return shlex.quote(str(text)) if quote else str(text)

        args = []
        if self.no_mmap:
            args.append("--no-mmap")
        args += [
            "-m",
            value(self.model_path),
            "--device",
            value(self.llama_device),
            "-ngl",
            str(self.n_gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
            "--ubatch-size",
            str(self.ubatch_size),
            "-fa",
            "on",
            "--poll",
            "1000",
            "-t",
            str(self.threads),
            "--cpu-mask",
            value(self.cpu_mask),
            "--cpu-strict",
            "1",
            "--no-display-prompt",
            "--no-warmup",
            "--single-turn",
            "--simple-io",
            "--temp",
            str(self.temperature),
            "-n",
            str(self.max_tokens if max_tokens_override is None else max_tokens_override),
            "-sys",
            value(system_prompt),
            "-f",
            value(prompt_file),
        ]
        if self.json_schema:
            args += ["-j", value(self.json_schema)]
        args.extend(value(arg) for arg in self.extra_args)
        return args

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        user_parts = [message["content"] for message in messages if message.get("role") != "system"]
        return "\n\n".join(user_parts)

    def _model_text(self, stdout: str) -> str:
        text = stdout.strip()
        text = re.sub(r"^system_info:.*?$", "", text, flags=re.MULTILINE).strip()
        marker = "Answer JSON only:"
        if marker in text:
            text = text.rsplit(marker, 1)[-1].strip()
        text = re.sub(r"\n\[ Prompt:[^\n]*\]\s*", "\n", text).strip()
        text = re.sub(r"\nExiting\.\.\.\s*$", "", text).strip()
        return text

    def _run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        timeout: int,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        if check and result.returncode != 0:
            raise LlamaCliError(
                json.dumps(
                    {
                        "returncode": result.returncode,
                        "stdout": result.stdout[-4000:],
                        "stderr": result.stderr[-4000:],
                        "command": command,
                    },
                    ensure_ascii=True,
                )
            )
        return result


def parse_llama_perf_timings(text: str) -> dict[str, Any]:
    timings: dict[str, Any] = {}
    patterns = {
        "load_ms": r"load time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        "sample_ms": r"sample time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        "prompt_ms": r"prompt eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        "decode_ms": r"eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
        "total_ms": r"total time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            timings[key] = float(match.group(1))
    token_patterns = {
        "prompt_tokens": r"prompt eval time\s*=.*?/\s*(\d+)\s*tokens",
        "decode_tokens": r"eval time\s*=.*?/\s*(\d+)\s*runs?",
    }
    for key, pattern in token_patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            timings[key] = int(match.group(1))
    if "prompt_ms" in timings or "decode_ms" in timings:
        timings["inference_wall_ms"] = float(timings.get("prompt_ms") or 0.0) + float(timings.get("decode_ms") or 0.0)
    if timings.get("prompt_ms") and timings.get("prompt_tokens"):
        timings["prompt_per_second"] = float(timings["prompt_tokens"]) / (float(timings["prompt_ms"]) / 1000.0)
    if timings.get("decode_ms") and timings.get("decode_tokens"):
        timings["decode_per_second"] = float(timings["decode_tokens"]) / (float(timings["decode_ms"]) / 1000.0)
    return timings


def build_prompt_payload(
    entrypoint: str,
    request: dict[str, Any],
    params: dict[str, Any],
    sample: dict[str, Any],
    tool_cfg: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "tool": entrypoint,
        "task": tool_task_description(entrypoint),
        "sample": sample,
        "parameters": params,
        "upstream": unwrap_upstream(request.get("upstream") or {}),
        "expected_schema": expected_schema(entrypoint),
    }
    if truthy(tool_cfg.get("allow_ground_truth_context")):
        payload["ground_truth_actions"] = sample.get("ground_truth_actions") or []
    return payload


def build_messages(entrypoint: str, prompt_payload: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are a phone-local inference tool for tau-bench workflow execution. "
        "Return only the answer JSON object for the requested tool. "
        "Do not repeat the input. Do not include markdown, prose, or code fences. "
        "The first character must be { and the last character must be }. "
        "Use empty strings or arrays when evidence is missing; do not invent IDs."
    )
    user = concise_user_prompt(entrypoint, prompt_payload)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def concise_user_prompt(entrypoint: str, prompt_payload: dict[str, Any]) -> str:
    sample = prompt_payload.get("sample") or {}
    params = prompt_payload.get("parameters") or {}
    upstream = prompt_payload.get("upstream") or {}
    instruction = str(sample.get("instruction") or "")
    common = [
        f"Tool: {entrypoint}",
        f"Domain hint: {sample.get('domain') or 'unknown'}",
        f"Sample id: {sample.get('experiment_id') or sample.get('task_id') or ''}",
        "User instruction:",
        instruction,
    ]
    if params:
        common += ["Parameters:", json.dumps(params, ensure_ascii=True)]
    if upstream:
        common += ["Upstream context:", json.dumps(upstream, ensure_ascii=True)[:6000]]

    templates = {
        "extract_tau_slots": {
            "domain": "",
            "intent": "",
            "customer": {"first_name": "", "last_name": "", "zip": "", "email": ""},
            "order_id": "",
            "reservation_id": "",
            "product_requests": [{"item_type": "", "constraints": [], "fallbacks": []}],
            "flight_requests": [{"origin": "", "destination": "", "date": "", "constraints": []}],
            "payment_preference": {"type": "", "identifier": "", "policy": ""},
            "needs_confirmation": False,
            "confidence": 0.0,
        },
        "match_retail_product_variant": {
            "matches": [
                {
                    "source_item_id": "",
                    "new_item_id": "",
                    "product_id": "",
                    "constraints_satisfied": [],
                    "reason": "",
                    "confidence": 0.0,
                }
            ],
            "unresolved": [],
        },
        "rank_airline_flight_options": {
            "ranked_options": [{"rank": 1, "flight_numbers": [], "dates": [], "score": 0.0, "reason": ""}],
            "constraints": {},
        },
        "detect_confirmation_or_revision": {
            "decision": "confirm|deny|revise|unclear",
            "revisions": {},
            "confidence": 0.0,
            "evidence": [],
        },
        "resolve_payment_preference": {
            "selected_payment_method_id": "",
            "payment_type": "",
            "allocation": [],
            "reason": "",
            "confidence": 0.0,
        },
    }
    guidance = {
        "extract_tau_slots": "Extract only facts present in the instruction.",
        "match_retail_product_variant": "Match only candidates present in upstream context or parameters.",
        "rank_airline_flight_options": "Rank only flights present in upstream context or parameters.",
        "detect_confirmation_or_revision": "Classify the user's current instruction as confirm, deny, revise, or unclear.",
        "resolve_payment_preference": "Select only a payment id present in upstream context or parameters.",
    }
    common += [
        "Instruction:",
        guidance[entrypoint],
        "Return this JSON shape with real values:",
        json.dumps(templates[entrypoint], ensure_ascii=True),
        "Answer JSON only:",
    ]
    return "\n".join(common)


def tool_task_description(entrypoint: str) -> str:
    descriptions = {
        "extract_tau_slots": "Extract structured slots from the user instruction.",
        "match_retail_product_variant": "Choose retail replacement item candidates that satisfy requested constraints.",
        "rank_airline_flight_options": "Rank airline flight or itinerary candidates against the user constraints.",
        "detect_confirmation_or_revision": "Classify whether the user confirms, denies, revises, or is unclear.",
        "resolve_payment_preference": "Choose the best payment method or allocation from available payment candidates.",
    }
    return descriptions[entrypoint]


def expected_schema(entrypoint: str) -> dict[str, Any]:
    schemas = {
        "extract_tau_slots": {
            "domain": "retail|airline|unknown",
            "intent": "short action label",
            "customer": {"first_name": "", "last_name": "", "zip": "", "email": ""},
            "order_id": "",
            "reservation_id": "",
            "product_requests": [{"item_type": "", "constraints": [], "fallbacks": []}],
            "flight_requests": [{"origin": "", "destination": "", "date": "", "constraints": []}],
            "payment_preference": {"type": "", "identifier": "", "policy": ""},
            "needs_confirmation": False,
            "confidence": 0.0,
        },
        "match_retail_product_variant": {
            "matches": [
                {
                    "source_item_id": "",
                    "new_item_id": "",
                    "product_id": "",
                    "constraints_satisfied": [],
                    "reason": "",
                    "confidence": 0.0,
                }
            ],
            "unresolved": [],
        },
        "rank_airline_flight_options": {
            "ranked_options": [
                {
                    "rank": 1,
                    "flight_numbers": [],
                    "dates": [],
                    "score": 0.0,
                    "reason": "",
                }
            ],
            "constraints": {},
        },
        "detect_confirmation_or_revision": {
            "decision": "confirm|deny|revise|unclear",
            "revisions": {},
            "confidence": 0.0,
            "evidence": [],
        },
        "resolve_payment_preference": {
            "selected_payment_method_id": "",
            "payment_type": "",
            "allocation": [],
            "reason": "",
            "confidence": 0.0,
        },
    }
    return schemas[entrypoint]


def resolve_sample(params: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if isinstance(params.get("sample"), dict):
        return sanitize_sample(params["sample"], include_ground_truth=truthy(cfg.get("allow_ground_truth_context")))

    sample_id = (
        params.get("sample_id")
        or params.get("experiment_id")
        or cfg.get("sample_id")
        or cfg.get("experiment_id")
    )
    inline_instruction = params.get("instruction") or cfg.get("instruction")
    if inline_instruction:
        return {
            "experiment_id": str(sample_id or ""),
            "domain": str(params.get("domain") or cfg.get("domain") or "unknown"),
            "instruction": str(inline_instruction),
        }

    dataset_path = Path(cfg.get("dataset_path") or params.get("dataset_path") or DEFAULT_DATASET_PATH)
    if not dataset_path.exists():
        return {"experiment_id": str(sample_id or ""), "domain": "unknown", "instruction": ""}
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else data
    samples = samples or []
    if sample_id:
        sample_id_text = str(sample_id)
        for sample in samples:
            if sample_id_text in {
                str(sample.get("experiment_id")),
                str(sample.get("sample_id")),
                str(sample.get("task_id")),
                str(sample.get("selection_index")),
            }:
                return sanitize_sample(sample, include_ground_truth=truthy(cfg.get("allow_ground_truth_context")))
    if samples:
        return sanitize_sample(samples[0], include_ground_truth=truthy(cfg.get("allow_ground_truth_context")))
    return {"experiment_id": str(sample_id or ""), "domain": "unknown", "instruction": ""}


def sanitize_sample(sample: dict[str, Any], *, include_ground_truth: bool) -> dict[str, Any]:
    allowed = {
        "experiment_id",
        "sample_id",
        "task_id",
        "selection_index",
        "domain",
        "instruction",
        "user_goal",
        "selection_reason",
        "inference_tool_coverage",
    }
    result = {key: sample.get(key) for key in allowed if key in sample}
    if "instruction" not in result:
        result["instruction"] = sample_instruction(sample)
    if include_ground_truth:
        result["ground_truth_actions"] = sample.get("ground_truth_actions") or []
    return result


def sample_instruction(sample: dict[str, Any]) -> str:
    if sample.get("instruction"):
        return str(sample["instruction"])
    if sample.get("user_goal"):
        return str(sample["user_goal"])
    instructions = ((sample.get("user_scenario") or {}).get("instructions") or {})
    return str(instructions.get("reason_for_call") or "")


def unwrap_upstream(upstream: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in upstream.items():
        if isinstance(value, dict) and "value" in value:
            result[key] = value["value"]
        else:
            result[key] = value
    return result


def extract_chat_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise LlamaServerError("llama-server response contained no choices")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        text = choices[0].get("text")
    if not text:
        raise LlamaServerError("llama-server response contained no content")
    return str(text)


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = strip_code_fence(text.strip())
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append((idx, end, obj))
    if candidates:
        substantial = [item for item in candidates if item[1] >= 80 or len(item[2]) >= 5]
        return (substantial or candidates)[-1][2]
    raise ValueError(f"Could not parse JSON object from llama output: {text[:300]}")


def strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_tool_output(entrypoint: str, parsed: dict[str, Any]) -> dict[str, Any]:
    if entrypoint == "extract_tau_slots":
        parsed.setdefault("domain", "unknown")
        parsed.setdefault("intent", "")
        parsed.setdefault("customer", {})
        parsed.setdefault("order_id", "")
        parsed.setdefault("reservation_id", "")
        parsed.setdefault("product_requests", [])
        parsed.setdefault("flight_requests", [])
        parsed.setdefault("payment_preference", {})
        parsed.setdefault("needs_confirmation", False)
        parsed.setdefault("confidence", 0.0)
        return parsed
    if entrypoint == "match_retail_product_variant":
        parsed.setdefault("matches", [])
        parsed.setdefault("unresolved", [])
        return parsed
    if entrypoint == "rank_airline_flight_options":
        parsed.setdefault("ranked_options", [])
        parsed.setdefault("constraints", {})
        return parsed
    if entrypoint == "detect_confirmation_or_revision":
        parsed.setdefault("decision", "unclear")
        parsed.setdefault("revisions", {})
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("evidence", [])
        return parsed
    if entrypoint == "resolve_payment_preference":
        parsed.setdefault("selected_payment_method_id", "")
        parsed.setdefault("payment_type", "")
        parsed.setdefault("allocation", [])
        parsed.setdefault("reason", "")
        parsed.setdefault("confidence", 0.0)
        return parsed
    return parsed


def apply_obvious_fallbacks(entrypoint: str, parsed: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if entrypoint != "extract_tau_slots":
        return parsed
    sample = payload.get("sample") or {}
    params = payload.get("parameters") or {}
    instruction = str(sample.get("instruction") or params.get("instruction") or "")
    if not parsed.get("domain") or parsed.get("domain") == "unknown":
        parsed["domain"] = infer_domain(instruction)
    if not parsed.get("intent"):
        parsed["intent"] = infer_intent(instruction)
    if not parsed.get("order_id"):
        match = re.search(r"#W\d+", instruction)
        if match:
            parsed["order_id"] = match.group(0)
    if not parsed.get("reservation_id"):
        match = re.search(r"\b[A-Z0-9]{6}\b", instruction)
        if match and not parsed.get("order_id"):
            parsed["reservation_id"] = match.group(0)
    payment = parsed.get("payment_preference") or {}
    if not payment.get("type"):
        parsed["payment_preference"] = infer_payment_preference(instruction)
    return parsed


def mock_tool_output(entrypoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    sample = payload.get("sample") or {}
    instruction = str(sample.get("instruction") or "")
    gt_actions = payload.get("ground_truth_actions") or []
    if not gt_actions:
        gt_actions = sample.get("ground_truth_actions") or []

    if entrypoint == "extract_tau_slots":
        name_match = re.search(r"\bYou are ([A-Z][A-Za-z'-]+) ([A-Z][A-Za-z'-]+)(?: in (\d{5}))?", instruction)
        order_match = re.search(r"#W\d+", instruction)
        reservation_match = re.search(r"\b[A-Z0-9]{6}\b", instruction)
        return {
            "domain": sample.get("domain") or infer_domain(instruction),
            "intent": infer_intent(instruction),
            "customer": {
                "first_name": name_match.group(1) if name_match else "",
                "last_name": name_match.group(2) if name_match else "",
                "zip": name_match.group(3) if name_match and name_match.group(3) else "",
                "email": "",
            },
            "order_id": order_match.group(0) if order_match else "",
            "reservation_id": reservation_match.group(0) if reservation_match and not order_match else "",
            "product_requests": extract_product_phrases(instruction),
            "flight_requests": [{"constraints": extract_constraint_phrases(instruction)}] if infer_domain(instruction) == "airline" else [],
            "payment_preference": infer_payment_preference(instruction),
            "needs_confirmation": bool(re.search(r"\b(confirm|make sure|detail-oriented|if there is no|would go)\b", instruction, re.I)),
            "confidence": 0.55,
        }
    if entrypoint == "match_retail_product_variant":
        mutation = first_action(gt_actions, {"exchange_delivered_order_items", "modify_pending_order_items"})
        args = mutation.get("arguments") if mutation else {}
        return {
            "matches": [
                {
                    "source_item_id": old,
                    "new_item_id": new,
                    "product_id": "",
                    "constraints_satisfied": extract_constraint_phrases(instruction),
                    "reason": "Mock mapping from available task hints.",
                    "confidence": 0.5,
                }
                for old, new in zip(args.get("item_ids") or [], args.get("new_item_ids") or [], strict=False)
            ],
            "unresolved": [],
        }
    if entrypoint == "rank_airline_flight_options":
        action = first_action(gt_actions, {"update_reservation_flights", "book_reservation", "search_direct_flight"})
        args = action.get("arguments") if action else {}
        flights = args.get("flights") or []
        return {
            "ranked_options": [
                {
                    "rank": idx + 1,
                    "flight_numbers": [flight.get("flight_number", "")] if isinstance(flight, dict) else [],
                    "dates": [flight.get("date", "")] if isinstance(flight, dict) else [],
                    "score": 1.0 - idx * 0.05,
                    "reason": "Mock ranking from task hints.",
                }
                for idx, flight in enumerate(flights)
            ],
            "constraints": {"phrases": extract_constraint_phrases(instruction)},
        }
    if entrypoint == "detect_confirmation_or_revision":
        decision = "revise" if re.search(r"\b(instead|change|if there is no|would go|but)\b", instruction, re.I) else "confirm"
        return {"decision": decision, "revisions": {}, "confidence": 0.5, "evidence": extract_constraint_phrases(instruction)}
    if entrypoint == "resolve_payment_preference":
        action = first_action(
            gt_actions,
            {
                "exchange_delivered_order_items",
                "modify_pending_order_items",
                "return_delivered_order_items",
                "update_reservation_flights",
                "update_reservation_baggages",
                "book_reservation",
            },
        )
        args = action.get("arguments") if action else {}
        payment_id = args.get("payment_method_id") or args.get("payment_id") or ""
        pref = infer_payment_preference(instruction)
        return {
            "selected_payment_method_id": payment_id,
            "payment_type": pref.get("type", ""),
            "allocation": [],
            "reason": "Mock payment resolution from instruction and task hints.",
            "confidence": 0.5,
        }
    raise ValueError(f"No mock implementation for {entrypoint}")


def first_action(actions: list[dict[str, Any]], names: set[str]) -> dict[str, Any]:
    return next((action for action in actions if action.get("name") in names), {})


def infer_domain(text: str) -> str:
    if re.search(r"\b(flight|airline|reservation|baggage|airport|cabin)\b", text, re.I):
        return "airline"
    if re.search(r"\b(order|item|product|return|exchange|payment)\b", text, re.I):
        return "retail"
    return "unknown"


def infer_intent(text: str) -> str:
    lowered = text.lower()
    for key in ["exchange", "return", "cancel", "book", "change", "modify", "update"]:
        if key in lowered:
            return key
    return "resolve_request"


def extract_product_phrases(text: str) -> list[dict[str, Any]]:
    phrases = []
    for item in re.findall(r"\b(?:keyboard|thermostat|lamp|camera|earbud|water bottle|mouse|machine)\b", text, re.I):
        phrases.append({"item_type": item.lower(), "constraints": extract_constraint_phrases(text), "fallbacks": []})
    return phrases


def extract_constraint_phrases(text: str) -> list[str]:
    patterns = [
        r"with [^.,;]+",
        r"compatible with [^.,;]+",
        r"cheapest",
        r"fastest",
        r"nonstop",
        r"direct",
        r"business",
        r"economy",
        r"lowest price",
        r"gift card",
        r"credit card",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, re.I):
            value = match.strip()
            if value and value.lower() not in {item.lower() for item in found}:
                found.append(value)
    return found


def infer_payment_preference(text: str) -> dict[str, str]:
    lowered = text.lower()
    if "gift card" in lowered:
        return {"type": "gift_card", "identifier": "", "policy": "prefer gift card"}
    if "credit card" in lowered:
        return {"type": "credit_card", "identifier": "", "policy": "prefer credit card"}
    if "certificate" in lowered:
        return {"type": "certificate", "identifier": "", "policy": "prefer certificate"}
    if "paypal" in lowered:
        return {"type": "paypal", "identifier": "", "policy": "prefer paypal"}
    return {"type": "", "identifier": "", "policy": ""}


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    compact = {key: response.get(key) for key in ("id", "model", "created", "object", "usage", "timings") if key in response}
    choices = response.get("choices") or []
    if choices:
        compact["finish_reason"] = choices[0].get("finish_reason")
    return compact


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
