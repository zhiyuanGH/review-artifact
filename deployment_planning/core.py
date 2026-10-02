"""Rule-based deployment planning for MOBench workflows."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from MOBench.deployment_planning import DEPLOYMENT_SCHEMA_VERSION, FAIR_CLI_DEPLOYMENT_STRATEGIES
    from MOBench.tools.cli import ALL_TOOLS, QNN_TOOLS
    from MOBench.tools.llm_tau import LLM_TAU_TOOLS
except ImportError:
    from deployment_planning import DEPLOYMENT_SCHEMA_VERSION, FAIR_CLI_DEPLOYMENT_STRATEGIES
    from tools.cli import ALL_TOOLS, QNN_TOOLS
    from tools.llm_tau import LLM_TAU_TOOLS


DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_LLM_MODEL = "llama-3.2-1b-q4_0"
DEFAULT_PHONE_MODEL = "/data/local/tmp/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf"
DEFAULT_PHONE_LLAMA_DIR = "/data/local/tmp/llama.cpp"
DEFAULT_ORT_GENAI_DIR = "/data/local/tmp/oga_generate"
DEFAULT_ORT_GENAI_MODEL = "/data/local/tmp/ort_llm_qwen_cpu"
DEFAULT_ORT_GENAI_JAR = "oga-generate.jar"
DEFAULT_GENIE_BUNDLE_DIR = "/data/local/tmp/genie_bundle"
FAIR_CLI_STRATEGY_SET = set(FAIR_CLI_DEPLOYMENT_STRATEGIES)
LEGACY_STRATEGY_ALIASES = {
    "cli-parallel-cpu": "cli-cpu-only",
    "cli-parallel-gpu": "cli-gpu-only",
    "cli-parallel-npu": "cli-npu-only",
}
CLI_DEVICE_SPECS = {
    "none": {"device_class": "cpu", "ngl": 0, "label": "CPU"},
    "GPUOpenCL": {"device_class": "gpu", "ngl": 99, "label": "GPU"},
    "HTP0": {"device_class": "npu", "ngl": 99, "label": "NPU"},
}


def build_runtime_defaults(args: Any, *, repo_root: Path) -> dict[str, Any]:
    adb = best_effort_adb_path(getattr(args, "adb", None))
    device = getattr(args, "adb_serial", None) or ""
    return {
        "max_parallel_nodes": int(getattr(args, "max_parallel_nodes", 4)),
        "python_executable": str((getattr(args, "python_executable", None) or default_python_executable(repo_root)).resolve()),
        "adb": str(adb) if adb else "",
        "adb_serial": device,
        "llm_url": getattr(args, "llm_url", DEFAULT_LLM_URL),
        "llm_model": getattr(args, "llm_model", DEFAULT_LLM_MODEL),
        "llm_max_tokens": int(getattr(args, "llm_max_tokens", 64)),
        "llm_temperature": float(getattr(args, "llm_temperature", 0.0)),
        "llm_timeout": int(getattr(args, "llm_timeout", 240)),
        "llama_health_url": getattr(args, "llama_health_url", None),
        "llama_local_port": getattr(args, "llama_local_port", None),
        "llama_remote_port": int(getattr(args, "llama_remote_port", 8080)),
        "llama_remote_dir": getattr(args, "llama_remote_dir", DEFAULT_PHONE_LLAMA_DIR),
        "llama_model_path": getattr(args, "llama_model_path", DEFAULT_PHONE_MODEL),
        "llama_server_device": getattr(args, "llama_server_device", "HTP0"),
        "llama_parallel": int(getattr(args, "llama_parallel", 2)),
        "llama_startup_timeout": int(getattr(args, "llama_startup_timeout", 240)),
        "llama_server_extra_args": list(getattr(args, "llama_server_extra_arg", []) or []),
        "llama_cli_timeout": int(getattr(args, "llama_cli_timeout", 240)),
        "llama_cli_extra_args": list(getattr(args, "llama_cli_extra_arg", []) or []),
        "llama_cli_warmup_runs": int(getattr(args, "llama_cli_warmup_runs", 1)),
        "llama_cli_warmup_max_tokens": int(getattr(args, "llama_cli_warmup_max_tokens", 1)),
        "llama_cli_warmup_prompt": getattr(args, "llama_cli_warmup_prompt", "OK"),
        "ort_genai_dir": getattr(args, "ort_genai_dir", DEFAULT_ORT_GENAI_DIR),
        "ort_genai_model_path": getattr(args, "ort_genai_model_path", DEFAULT_ORT_GENAI_MODEL),
        "ort_genai_jar": getattr(args, "ort_genai_jar", DEFAULT_ORT_GENAI_JAR),
        "ort_genai_timeout": int(getattr(args, "ort_genai_timeout", getattr(args, "llm_timeout", 240))),
        "ort_genai_prompt_char_limit": int(getattr(args, "ort_genai_prompt_char_limit", 512)),
        "genie_bundle_dir": getattr(args, "genie_bundle_dir", DEFAULT_GENIE_BUNDLE_DIR),
        "genie_config": getattr(args, "genie_config", "auto"),
        "genie_backend": getattr(args, "genie_backend", "npu"),
        "genie_log_level": getattr(args, "genie_log_level", "warn"),
        "genie_timeout": int(getattr(args, "genie_timeout", getattr(args, "llm_timeout", 240))),
    }


def build_deployment_document(
    workflow_doc: dict[str, Any],
    *,
    strategy: str,
    run_dir: Path,
    planning_workflow_path: Path,
    selected_samples_path: Path | None,
    runtime_defaults: dict[str, Any],
) -> dict[str, Any]:
    deployment_plan = build_deployment_plan(
        workflow_doc,
        strategy=strategy,
        run_dir=run_dir,
        planning_workflow_path=planning_workflow_path,
        selected_samples_path=selected_samples_path,
        runtime_defaults=runtime_defaults,
    )
    deployment_doc = {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "planning_workflow_json": str(planning_workflow_path),
        "selected_samples": str(selected_samples_path) if selected_samples_path else None,
        "workflow_document": workflow_doc,
        "deployment_plan": deployment_plan,
        "deployment_metadata": {
            "strategy": strategy,
            "canonical_strategy": canonical_strategy(strategy),
            "fair_comparison": canonical_strategy(strategy) in FAIR_CLI_STRATEGY_SET,
            "deployment_is_authoritative": True,
            "planning_preferred_backend_is_hint_only": True,
        },
    }
    validate_deployment_document(deployment_doc)
    return deployment_doc


def build_deployment_plan(
    workflow_doc: dict[str, Any],
    *,
    strategy: str,
    run_dir: Path,
    planning_workflow_path: Path,
    selected_samples_path: Path | None,
    runtime_defaults: dict[str, Any],
) -> dict[str, Any]:
    canonical = canonical_strategy(strategy)
    source_workflows = workflow_doc.get("workflows") or []
    global_round_robin_devices = (
        global_round_robin_llm_device_assignments(source_workflows)
        if canonical == "cli-balanced-round-robin"
        else {}
    )
    workflows = []
    for workflow_index, workflow in enumerate(source_workflows):
        node_deployments = deploy_workflow_nodes(
            workflow,
            strategy=strategy,
            runtime_defaults=runtime_defaults,
            llm_device_overrides=global_round_robin_devices.get(workflow_index),
        )
        workflows.append(
            {
                "workflow_index": workflow_index,
                "workflow_id": workflow.get("workflow_id"),
                "name": workflow.get("name"),
                "node_deployments": node_deployments,
                "concurrency_groups": concurrency_groups(workflow, node_deployments),
            }
        )

    return {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "strategy": strategy,
        "canonical_strategy": canonical,
        "fair_comparison": canonical in FAIR_CLI_STRATEGY_SET,
        "source_workflow_json": str(planning_workflow_path),
        "selected_samples": str(selected_samples_path) if selected_samples_path else None,
        "run_dir": str(run_dir),
        "runtime_defaults": sanitize_runtime_defaults(runtime_defaults),
        "workflows": workflows,
    }


def deploy_node(node: dict[str, Any], *, strategy: str, runtime_defaults: dict[str, Any]) -> dict[str, Any]:
    return deploy_node_with_context(
        node,
        strategy=strategy,
        runtime_defaults=runtime_defaults,
        llm_device=None,
    )


def deploy_workflow_nodes(
    workflow: dict[str, Any],
    *,
    strategy: str,
    runtime_defaults: dict[str, Any],
    llm_device_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    canonical = canonical_strategy(strategy)
    llm_devices = (
        llm_device_overrides
        if llm_device_overrides is not None
        else llm_device_assignments(workflow, canonical, runtime_defaults=runtime_defaults)
    )
    return [
        deploy_node_with_context(
            node,
            strategy=strategy,
            runtime_defaults=runtime_defaults,
            llm_device=llm_devices.get(node.get("node_id")),
        )
        for node in workflow.get("nodes") or []
    ]


def deploy_node_with_context(
    node: dict[str, Any],
    *,
    strategy: str,
    runtime_defaults: dict[str, Any],
    llm_device: str | None,
) -> dict[str, Any]:
    canonical = canonical_strategy(strategy)
    fair_comparison = canonical in FAIR_CLI_STRATEGY_SET
    kind = node.get("kind", "app")
    entrypoint = node.get("entrypoint") or node.get("node_id")
    preferred = node.get("preferred_backend") or "unknown"
    if canonical == "mock":
        return node_deployment(
            node,
            algorithm_name=strategy,
            fair_comparison=False,
            selected_runtime="mock",
            device_class="mock",
            device="mock",
            ngl=None,
            concurrency_group="mock",
            reason="Mock deployment strategy maps every node to deterministic mock output.",
            concrete={"mode": "mock"},
        )

    if kind == "llm":
        if canonical == "server-batch-npu":
            return node_deployment(
                node,
                algorithm_name=strategy,
                fair_comparison=False,
                selected_runtime="llama_server",
                device_class="npu",
                device=runtime_defaults["llama_server_device"],
                ngl=99,
                concurrency_group="shared_llama_server",
                reason="LLM nodes share one phone llama-server with continuous batching on the NPU.",
                concrete={
                    "url": runtime_defaults["llm_url"],
                    "model": runtime_defaults["llm_model"],
                    "device": runtime_defaults["llama_server_device"],
                    "parallel": runtime_defaults["llama_parallel"],
                    "ngl": 99,
                },
            )
        if canonical == "framework-ort-genai-cpu":
            return node_deployment(
                node,
                algorithm_name=strategy,
                fair_comparison=False,
                selected_runtime="ort_genai_cli",
                device_class="cpu",
                device="none",
                ngl=None,
                concurrency_group="parallel_ort_genai_cpu",
                reason="Framework characterization: ORT GenAI package is CPU-only, so every LLM node in each wave is constrained to CPU.",
                concrete={
                    "model_path": runtime_defaults["ort_genai_model_path"],
                    "runner_dir": runtime_defaults["ort_genai_dir"],
                    "jar": runtime_defaults["ort_genai_jar"],
                    "device": "none",
                },
            )
        if canonical == "framework-genie-npu":
            return node_deployment(
                node,
                algorithm_name=strategy,
                fair_comparison=False,
                selected_runtime="genie_cli",
                device_class="npu",
                device="HTP0",
                ngl=None,
                concurrency_group="parallel_genie_npu",
                reason="Framework characterization: current Genie bundle is NPU-only, so every LLM node in each wave is constrained to NPU.",
                concrete={
                    "bundle_dir": runtime_defaults["genie_bundle_dir"],
                    "config": runtime_defaults["genie_config"],
                    "backend": runtime_defaults["genie_backend"],
                    "device": "HTP0",
                },
            )
        if canonical not in FAIR_CLI_STRATEGY_SET:
            raise ValueError(f"Unsupported deployment strategy for LLM node: {strategy}")
        device = llm_device or "HTP0"
        spec = CLI_DEVICE_SPECS[device]
        device_class = spec["device_class"]
        ngl = spec["ngl"]
        return node_deployment(
            node,
            algorithm_name=strategy,
            fair_comparison=fair_comparison,
            selected_runtime="llama_cli",
            device_class=device_class,
            device=device,
            ngl=ngl,
            concurrency_group=llama_cli_concurrency_group(canonical, device_class),
            reason=llama_cli_reason(canonical, device, strategy),
            concrete={
                "model_path": runtime_defaults["llama_model_path"],
                "device": device,
                "ngl": ngl,
            },
        )

    if kind == "dnn" or entrypoint in QNN_TOOLS:
        return node_deployment(
            node,
            algorithm_name=strategy,
            fair_comparison=fair_comparison,
            selected_runtime="json_cli",
            device_class="npu",
            device="qnn_or_script_adapter",
            ngl=None,
            concurrency_group="tool_json_cli",
            reason="DNN/QNN-style tools are invoked through the JSON CLI adapter and keep NPU as the placement class.",
            concrete={"mode": "json_cli", "entrypoint": entrypoint},
        )

    if entrypoint in LLM_TAU_TOOLS:
        return node_deployment(
            node,
            algorithm_name=strategy,
            fair_comparison=fair_comparison,
            selected_runtime="json_cli_llama_cli_tool",
            device_class="cpu",
            device="none",
            ngl=0,
            concurrency_group="tool_json_cli",
            reason="Semantic helper tools remain logical tools, implemented by phone llama-cli on CPU by default.",
            concrete={"mode": "json_cli", "llm_runtime": "llama_cli", "llama_device": "none", "ngl": 0},
        )

    return node_deployment(
        node,
        algorithm_name=strategy,
        fair_comparison=fair_comparison,
        selected_runtime="json_cli_script",
        device_class="cpu",
        device="host_python",
        ngl=None,
        concurrency_group="tool_json_cli",
        reason=f"Tool node uses deterministic script/json CLI backend; preferred_backend={preferred!r} is treated as a hint only.",
        concrete={"mode": "json_cli", "entrypoint": entrypoint},
    )


def canonical_strategy(strategy: str) -> str:
    return LEGACY_STRATEGY_ALIASES.get(strategy, strategy)


def llm_device_assignments(
    workflow: dict[str, Any],
    canonical_strategy_name: str,
    *,
    runtime_defaults: dict[str, Any] | None = None,
) -> dict[str, str]:
    node_map = {node.get("node_id"): node for node in workflow.get("nodes") or []}
    llm_nodes = [
        str(node.get("node_id"))
        for node in workflow.get("nodes") or []
        if node.get("kind") == "llm" and node.get("node_id")
    ]
    if not llm_nodes:
        return {}
    if canonical_strategy_name == "cli-cpu-only":
        return {node_id: "none" for node_id in llm_nodes}
    if canonical_strategy_name == "cli-gpu-only":
        return {node_id: "GPUOpenCL" for node_id in llm_nodes}
    if canonical_strategy_name == "cli-npu-only":
        return {node_id: "HTP0" for node_id in llm_nodes}
    if canonical_strategy_name == "framework-ort-genai-cpu":
        return {node_id: "none" for node_id in llm_nodes}
    if canonical_strategy_name == "framework-genie-npu":
        return {node_id: "HTP0" for node_id in llm_nodes}
    if canonical_strategy_name == "cli-balanced-round-robin":
        order = ["none", "GPUOpenCL", "HTP0"]
        return {node_id: order[index % len(order)] for index, node_id in enumerate(llm_nodes)}
    if canonical_strategy_name == "cli-odabench-wo-decider":
        assignments = llm_device_assignments(
            workflow,
            "cli-balanced-wave-aware",
            runtime_defaults=runtime_defaults,
        )
        best_device = odabench_wo_decider_single_u_device(runtime_defaults)
        for wave in execution_waves_for_deployment(workflow):
            wave_llms = [
                node_id
                for node_id in wave.get("node_ids", [])
                if node_map.get(node_id, {}).get("kind") == "llm"
            ]
            if len(wave_llms) < 2:
                continue
            for node_id in wave_llms:
                assignments[node_id] = best_device
        for node_id in llm_nodes:
            assignments.setdefault(node_id, best_device)
        return assignments
    if canonical_strategy_name != "cli-balanced-wave-aware":
        return {}

    assignments: dict[str, str] = {}
    wave_order = wave_aware_llm_device_order(runtime_defaults)
    for wave in execution_waves_for_deployment(workflow):
        wave_llms = [
            node_id
            for node_id in wave.get("node_ids", [])
            if node_map.get(node_id, {}).get("kind") == "llm"
        ]
        if len(wave_llms) == 1:
            order = wave_order[:1]
        elif len(wave_llms) == 2:
            order = wave_order[:2]
        else:
            order = wave_order
        for index, node_id in enumerate(wave_llms):
            assignments[node_id] = order[index % len(order)]
    for node_id in llm_nodes:
        assignments.setdefault(node_id, wave_order[0])
    return assignments


def wave_aware_llm_device_order(runtime_defaults: dict[str, Any] | None = None) -> list[str]:
    """Return empirical best-to-worst llama.cpp device order for wave placement."""
    model_path = str((runtime_defaults or {}).get("llama_model_path") or "").lower()
    device_serial = str((runtime_defaults or {}).get("adb_serial") or "")
    if "qwen3-0.6b" in model_path or "qwen3_0_6b" in model_path:
        base_order = ["none", "HTP0", "GPUOpenCL"]
    elif "gemma-3-1b" in model_path or "gemma_3_1b" in model_path:
        base_order = ["none", "HTP0", "GPUOpenCL"]
    elif "llama-3.2-3b" in model_path or "llama_3_2_3b" in model_path:
        if device_serial == os.environ.get("MOBENCH_NPU_FIRST_LLAMA3B_SERIAL", "<device-serial>"):
            base_order = ["HTP0", "none", "GPUOpenCL"]
        else:
            base_order = ["none", "HTP0", "GPUOpenCL"]
    elif "llama-3.2-1b" in model_path or "llama_3_2_1b" in model_path:
        base_order = ["HTP0", "none", "GPUOpenCL"]
    elif "qwen1_5-1_8b" in model_path or "qwen1_5_1_8b" in model_path:
        base_order = ["HTP0", "none", "GPUOpenCL"]
    else:
        base_order = ["HTP0", "none", "GPUOpenCL"]
    max_parallel_nodes = int((runtime_defaults or {}).get("max_parallel_nodes") or len(base_order))
    return base_order[: max(1, min(len(base_order), max_parallel_nodes))]


def odabench_wo_decider_single_u_device(runtime_defaults: dict[str, Any] | None = None) -> str:
    """Return the best single-U device for ODABench-wo-decider.

    This ablation keeps ODABench placements for serial LLM waves, but collapses
    concurrent LLM waves onto the empirically best single-U device for the
    device+model case. The current completed single-U matrix picks CPU for
    Qwen3 0.6B and NPU for Llama 3.2 1B / Qwen1.5 1.8B on both phones.
    """
    model_path = str((runtime_defaults or {}).get("llama_model_path") or "").lower()
    if "qwen3-0.6b" in model_path or "qwen3_0_6b" in model_path:
        return "none"
    if "llama-3.2-1b" in model_path or "llama_3_2_1b" in model_path:
        return "HTP0"
    if "qwen1_5-1_8b" in model_path or "qwen1_5_1_8b" in model_path:
        return "HTP0"
    return wave_aware_llm_device_order(runtime_defaults)[0]


def global_round_robin_llm_device_assignments(workflows: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """Assign LLM nodes across all workflows without resetting at workflow boundaries."""
    order = ["none", "GPUOpenCL", "HTP0"]
    next_index = 0
    assignments_by_workflow: dict[int, dict[str, str]] = {}
    for workflow_index, workflow in enumerate(workflows):
        workflow_assignments: dict[str, str] = {}
        for node in workflow.get("nodes") or []:
            node_id = node.get("node_id")
            if node.get("kind") != "llm" or not node_id:
                continue
            workflow_assignments[str(node_id)] = order[next_index % len(order)]
            next_index += 1
        assignments_by_workflow[workflow_index] = workflow_assignments
    return assignments_by_workflow


def execution_waves_for_deployment(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    waves = workflow.get("execution_hints", {}).get("parallel_waves") or []
    if waves:
        return sorted(waves, key=lambda item: int(item.get("wave_index", 0)))
    return topological_waves_for_deployment(workflow)


def topological_waves_for_deployment(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    node_ids = [node["node_id"] for node in workflow.get("nodes", []) if node.get("node_id")]
    outgoing = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in workflow.get("edges", []):
        src = edge.get("from_node")
        dst = edge.get("to_node")
        if src in outgoing and dst in indegree:
            outgoing[src].append(dst)
            indegree[dst] += 1
    ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
    waves: list[dict[str, Any]] = []
    visited = 0
    while ready:
        current = ready
        ready = []
        waves.append(
            {
                "wave_index": len(waves),
                "node_ids": current,
                "reason": "Derived by deployment planner for placement.",
            }
        )
        for node_id in current:
            visited += 1
            for dst in outgoing[node_id]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
    if visited != len(node_ids):
        raise ValueError(f"Workflow has a cycle and cannot be deployed: {workflow.get('workflow_id')}")
    return waves


def llama_cli_concurrency_group(canonical: str, device_class: str) -> str:
    if canonical in {"cli-balanced-round-robin", "cli-balanced-wave-aware", "cli-odabench-wo-decider"}:
        return f"parallel_llama_cli_balanced_{device_class}"
    return f"parallel_llama_cli_{device_class}"


def llama_cli_reason(canonical: str, device: str, strategy: str) -> str:
    spec = CLI_DEVICE_SPECS[device]
    label = spec["label"]
    if canonical == "cli-balanced-round-robin":
        return f"Fair CLI benchmark: global round-robin assigns this LLM node to {label} as a separate phone llama-cli process."
    if canonical == "cli-balanced-wave-aware":
        return f"Fair CLI benchmark: model-aware wave placement assigns this LLM node to {label}, using CPU/NPU before GPU when concurrent nodes need to be spread."
    if canonical == "cli-odabench-wo-decider":
        return f"ODABench-wo-decider ablation: serial LLM waves keep ODABench placement, while parallel LLM waves collapse to the best single-U device ({label}) for this model."
    if strategy != canonical:
        return f"Legacy alias {strategy!r} maps this LLM node to {label} as a separate phone llama-cli process."
    return f"Fair CLI benchmark: every LLM node runs as its own phone llama-cli process on {label}."


def node_deployment(
    node: dict[str, Any],
    *,
    algorithm_name: str,
    fair_comparison: bool,
    selected_runtime: str,
    device_class: str,
    device: str,
    ngl: int | None,
    concurrency_group: str,
    reason: str,
    concrete: dict[str, Any],
) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "kind": node.get("kind", "app"),
        "entrypoint": node.get("entrypoint") or node.get("node_id"),
        "label": node.get("label"),
        "preferred_backend_hint": node.get("preferred_backend") or "unknown",
        "algorithm_name": algorithm_name,
        "fair_comparison": fair_comparison,
        "selected_runtime": selected_runtime,
        "device_class": device_class,
        "device": device,
        "ngl": ngl,
        "concurrency_group": concurrency_group,
        "placement_reason": reason,
        "concrete_settings": concrete,
    }


def concurrency_groups(workflow: dict[str, Any], deployments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[str]] = {}
    for deployment in deployments:
        by_group.setdefault(deployment["concurrency_group"], []).append(deployment["node_id"])
    waves = workflow.get("execution_hints", {}).get("parallel_waves") or []
    return [
        {
            "group_id": group_id,
            "node_ids": node_ids,
            "wave_ids": [
                wave.get("wave_index")
                for wave in waves
                if set(wave.get("node_ids") or []) & set(node_ids)
            ],
        }
        for group_id, node_ids in sorted(by_group.items())
    ]


def build_runtime_config(
    workflow: dict[str, Any],
    workflow_deployment: dict[str, Any],
    *,
    runtime_defaults: dict[str, Any],
    dataset_path: Path | None,
    state_path: Path,
) -> dict[str, Any]:
    deployments = workflow_deployment.get("node_deployments") or []
    config: dict[str, Any] = {
        "max_parallel_nodes": runtime_defaults["max_parallel_nodes"],
        "deployment": {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "strategy": workflow_deployment.get("strategy"),
            "workflow_id": workflow.get("workflow_id"),
            "node_deployments": deployments,
        },
    }
    config["llm"] = llm_runtime_config(deployments, runtime_defaults)
    config["qnn_tools"] = tool_runtime_configs(
        workflow,
        deployments,
        runtime_defaults=runtime_defaults,
        dataset_path=dataset_path,
        state_path=state_path,
    )
    return config


def llm_runtime_config(deployments: list[dict[str, Any]], runtime_defaults: dict[str, Any]) -> dict[str, Any]:
    llm_deployments = [item for item in deployments if item.get("kind") == "llm"]
    if not llm_deployments:
        return {
            "runtime": "llama_server",
            "url": runtime_defaults["llm_url"],
            "model": runtime_defaults["llm_model"],
            "max_tokens": runtime_defaults["llm_max_tokens"],
            "temperature": runtime_defaults["llm_temperature"],
            "timeout": runtime_defaults["llm_timeout"],
        }
    if all(item["selected_runtime"] == "mock" for item in llm_deployments):
        return {"runtime": "mock"}
    if all(item["selected_runtime"] == "ort_genai_cli" for item in llm_deployments):
        return {
            "runtime": "ort_genai_cli",
            "max_tokens": runtime_defaults["llm_max_tokens"],
            "temperature": runtime_defaults["llm_temperature"],
            "timeout": runtime_defaults["ort_genai_timeout"],
            "ort_genai": ort_genai_config(runtime_defaults),
        }
    if all(item["selected_runtime"] == "genie_cli" for item in llm_deployments):
        return {
            "runtime": "genie_cli",
            "max_tokens": runtime_defaults["llm_max_tokens"],
            "temperature": runtime_defaults["llm_temperature"],
            "timeout": runtime_defaults["genie_timeout"],
            "genie": genie_config(runtime_defaults),
        }
    if any(item["selected_runtime"] == "llama_cli" for item in llm_deployments):
        first = next(item for item in llm_deployments if item["selected_runtime"] == "llama_cli")
        device = first["device"]
        overrides = {
            item["node_id"]: llama_cli_config(runtime_defaults, item["device"])
            for item in llm_deployments
            if item["selected_runtime"] == "llama_cli"
        }
        return {
            "runtime": "llama_cli",
            "max_tokens": runtime_defaults["llm_max_tokens"],
            "temperature": runtime_defaults["llm_temperature"],
            "timeout": runtime_defaults["llm_timeout"],
            "llama_cli": llama_cli_config(runtime_defaults, device),
            "llama_cli_overrides": overrides,
        }
    return {
        "runtime": "llama_server",
        "url": runtime_defaults["llm_url"],
        "model": runtime_defaults["llm_model"],
        "max_tokens": runtime_defaults["llm_max_tokens"],
        "temperature": runtime_defaults["llm_temperature"],
        "timeout": runtime_defaults["llm_timeout"],
    }


def tool_runtime_configs(
    workflow: dict[str, Any],
    deployments: list[dict[str, Any]],
    *,
    runtime_defaults: dict[str, Any],
    dataset_path: Path | None,
    state_path: Path,
) -> dict[str, Any]:
    deployment_by_entrypoint = {
        item["entrypoint"]: item
        for item in deployments
        if item.get("kind", "app") in {"app", "dnn"}
    }
    entrypoints = sorted(
        {
            str(node.get("entrypoint") or node.get("node_id"))
            for node in workflow.get("nodes", [])
            if node.get("kind", "app") in {"app", "dnn"}
        }
    )
    configs: dict[str, Any] = {}
    for entrypoint in entrypoints:
        deployment = deployment_by_entrypoint.get(entrypoint, {})
        if deployment.get("selected_runtime") == "mock":
            configs[entrypoint] = {"mode": "mock", "deployment": deployment}
            continue
        if dataset_path is None:
            raise FileNotFoundError("Deployment planning needs planning/selected_samples.json for json_cli tools.")
        extra_tool_config = {}
        if entrypoint in LLM_TAU_TOOLS:
            extra_tool_config = {
                "llm_runtime": "llama_cli",
                **llama_cli_config(runtime_defaults, "none"),
                "json_schema": "{}",
            }
        tool_timeout = max(120, int(runtime_defaults["llama_cli_timeout"]) + 60) if extra_tool_config else 120
        configs[entrypoint] = {
            "mode": "json_cli",
            "command": [
                runtime_defaults["python_executable"],
                "-m",
                "MOBench.tools.cli",
                entrypoint,
                "--dataset",
                str(dataset_path),
                "--state",
                str(state_path),
            ],
            "cwd": str(Path(__file__).resolve().parents[2]),
            "timeout": tool_timeout,
            "deployment": deployment,
            **extra_tool_config,
        }
    return configs


def llama_cli_config(runtime_defaults: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        "llama_cli_target": "phone",
        "adb": runtime_defaults.get("adb", ""),
        "device_serial": runtime_defaults.get("adb_serial", ""),
        "remote_dir": runtime_defaults["llama_remote_dir"],
        "model_path": runtime_defaults["llama_model_path"],
        "llama_device": device,
        "ngl": 0 if device == "none" else 99,
        "hexagon_env": device.startswith("HTP"),
        "max_tokens": runtime_defaults["llm_max_tokens"],
        "temperature": runtime_defaults["llm_temperature"],
        "llama_timeout": runtime_defaults["llama_cli_timeout"],
        "extra_args": runtime_defaults["llama_cli_extra_args"],
        "warmup_runs": runtime_defaults.get("llama_cli_warmup_runs", 1),
        "warmup_max_tokens": runtime_defaults.get("llama_cli_warmup_max_tokens", 1),
        "warmup_prompt": runtime_defaults.get("llama_cli_warmup_prompt", "OK"),
        "backend": "llama_cli_phone",
    }


def ort_genai_config(runtime_defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        "adb": runtime_defaults.get("adb", ""),
        "device_serial": runtime_defaults.get("adb_serial", ""),
        "runner_dir": runtime_defaults["ort_genai_dir"],
        "jar": runtime_defaults["ort_genai_jar"],
        "model_path": runtime_defaults["ort_genai_model_path"],
        "max_tokens": runtime_defaults["llm_max_tokens"],
        "temperature": runtime_defaults["llm_temperature"],
        "timeout": runtime_defaults["ort_genai_timeout"],
        "prompt_char_limit": runtime_defaults["ort_genai_prompt_char_limit"],
        "backend": "ort_genai_phone",
    }


def genie_config(runtime_defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        "adb": runtime_defaults.get("adb", ""),
        "device_serial": runtime_defaults.get("adb_serial", ""),
        "bundle_dir": runtime_defaults["genie_bundle_dir"],
        "config": runtime_defaults["genie_config"],
        "backend": runtime_defaults["genie_backend"],
        "log_level": runtime_defaults["genie_log_level"],
        "max_tokens": runtime_defaults["llm_max_tokens"],
        "temperature": runtime_defaults["llm_temperature"],
        "timeout": runtime_defaults["genie_timeout"],
    }


def workflow_deployment_for_index(deployment_plan: dict[str, Any], workflow_index: int) -> dict[str, Any]:
    for item in deployment_plan.get("workflows") or []:
        if int(item.get("workflow_index", -1)) == workflow_index:
            result = dict(item)
            result["strategy"] = deployment_plan.get("strategy")
            return result
    raise IndexError(f"No deployment record for workflow index {workflow_index}")


def deployment_requires_llama_server(deployment_plan: dict[str, Any], selected_indices: list[int]) -> bool:
    selected = set(selected_indices)
    for workflow in deployment_plan.get("workflows") or []:
        if int(workflow.get("workflow_index", -1)) not in selected:
            continue
        for node in workflow.get("node_deployments") or []:
            if node.get("selected_runtime") == "llama_server":
                return True
            if node.get("concrete_settings", {}).get("llm_runtime") == "llama_server":
                return True
    return False


def validate_deployment_document(deployment_doc: dict[str, Any]) -> None:
    if deployment_doc.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION:
        raise ValueError(f"Deployment document must use schema_version {DEPLOYMENT_SCHEMA_VERSION}.")
    workflow_doc = deployment_doc.get("workflow_document")
    deployment_plan = deployment_doc.get("deployment_plan")
    if not isinstance(workflow_doc, dict) or not isinstance(deployment_plan, dict):
        raise ValueError("Deployment document must contain workflow_document and deployment_plan objects.")
    plan_workflows = deployment_plan.get("workflows") or []
    workflows = workflow_doc.get("workflows") or []
    if len(plan_workflows) != len(workflows):
        raise ValueError(f"Deployment plan has {len(plan_workflows)} workflows for {len(workflows)} planned workflows.")
    for workflow_index, workflow in enumerate(workflows):
        node_ids = {node.get("node_id") for node in workflow.get("nodes") or []}
        deployed = workflow_deployment_for_index(deployment_plan, workflow_index)
        deployed_ids = {item.get("node_id") for item in deployed.get("node_deployments") or []}
        missing = sorted(node_ids - deployed_ids)
        extra = sorted(deployed_ids - node_ids)
        if missing or extra:
            raise ValueError(
                f"Deployment mismatch for workflow {workflow_index}: missing={missing} extra={extra}"
            )
        for item in deployed.get("node_deployments") or []:
            if item.get("entrypoint") and item.get("kind") in {"app", "dnn"} and item["entrypoint"] not in ALL_TOOLS:
                raise ValueError(f"Unknown tool entrypoint in deployment: {item['entrypoint']}")


def sanitize_runtime_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(defaults, ensure_ascii=True))


def best_effort_adb_path(explicit_path: Path | None) -> Path | None:
    if explicit_path:
        return explicit_path.resolve()
    env_adb = os.environ.get("ADB")
    if env_adb and Path(env_adb).exists():
        return Path(env_adb).resolve()
    found = shutil.which("adb")
    return Path(found).resolve() if found else None


def default_python_executable(repo_root: Path) -> Path:
    for venv_python in (repo_root / ".venv" / "bin" / "python", repo_root / ".venv" / "Scripts" / "python.exe"):
        if venv_python.exists():
            return venv_python
    return Path(sys.executable)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
