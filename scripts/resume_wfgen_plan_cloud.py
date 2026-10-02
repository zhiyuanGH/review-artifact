"""Resume per-sample cloud MOBench planning without redoing completed requests."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
for import_root in (PACKAGE_PARENT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from MOBench.workflow_generation import DEFAULT_QWEN_API_KEY_ENV, DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL  # noqa: E402
from MOBench.workflow_generation.artifacts import write_json  # noqa: E402
from MOBench.workflow_generation.cloud_llm import CloudLLMConfig  # noqa: E402
from MOBench.workflow_generation.json_generator import (  # noqa: E402
    generate_cloud_workflow_json,
    merge_workflow_documents,
    parse_workflow_json,
)
from MOBench.workflow_generation.sample_selector import select_tau_bench_samples  # noqa: E402
from MOBench.workflow_generation.workflow_validator import validate_cloud_workflow  # noqa: E402
from MOBench.workflow_planning.postprocess import hydrate_tool_parameters_from_samples, insert_llm_checker_nodes  # noqa: E402

import wfgen_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "planning"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_payload, selection = select_tau_bench_samples(
        sample_count=args.sample_count,
        selection_strategy=args.selection_strategy,
        sample_offset=args.sample_offset,
        seed=args.seed,
        include_empty_actions=args.include_empty_actions,
        tau_repo=args.tau_repo,
        tau_config=args.tau_config,
        tau_split=args.tau_split,
        selected_samples=args.selected_samples,
    )
    samples = selected_payload.get("samples") or []
    selected_path = output_dir / "selected_samples.json"
    write_json(selected_path, selected_payload)

    prompt_args = prompt_namespace(args)
    combined_prompt = wfgen_plan.build_combined_prompt_bundle(selected_payload, prompt_args)
    combined_prompt = wfgen_plan.write_planning_prompt_artifacts(combined_prompt, output_dir, prompt_args)

    cloud_config = build_cloud_config(args)
    generation_artifacts: dict[str, Any] = {
        "resume": True,
        "resumed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cloud_config": cloud_config.metadata(),
        "selected_samples": str(selected_path),
        "prompts": combined_prompt.artifacts,
        "samples": [],
    }

    per_sample_docs: list[dict[str, Any]] = []
    reused_count = 0
    generated_count = 0
    for sample_idx, sample in enumerate(samples):
        sample_label = wfgen_plan.sample_artifact_label(sample_idx, sample)
        sample_dir = output_dir / "samples" / sample_label
        sample_payload = sample_payload_for(selected_payload, sample)
        prompt = wfgen_plan.build_sample_prompt(selected_payload, sample, prompt_args)
        prompt = wfgen_plan.write_planning_prompt_artifacts(prompt, sample_dir, prompt_args)
        raw_path = sample_dir / "llm" / "raw_response.json"
        call_label = f"sample {sample_idx + 1}/{len(samples)} selection_index={sample.get('selection_index')}"

        existing_doc = load_existing_sample_doc(
            raw_path,
            sample_payload=sample_payload,
            cloud_config=cloud_config,
            insert_checker_nodes=not args.no_insert_llm_checker_nodes,
        )
        if existing_doc is not None:
            reused_count += 1
            sample_doc = existing_doc
            sample_artifacts = {"raw_response": str(raw_path), "reused_raw_response": True}
            print(
                f"[resume-plan] reused raw response [{call_label}] {raw_path}",
                flush=True,
            )
        else:
            generated_count += 1
            print(
                f"[resume-plan] generating missing sample [{call_label}] "
                f"user_prompt_chars={len(prompt.user_prompt_text)} catalog_entries={len(prompt.tool_list)}",
                flush=True,
            )
            sample_doc, sample_artifacts = generate_cloud_workflow_json(
                config=cloud_config,
                prompt=prompt,
                selected_payload=sample_payload,
                output_dir=sample_dir,
                insert_checker_nodes=not args.no_insert_llm_checker_nodes,
                call_label=call_label,
            )
        per_sample_docs.append(sample_doc)
        generation_artifacts["samples"].append(
            {
                "sample_index": sample_idx,
                "selection_index": sample.get("selection_index"),
                "task_id": sample.get("task_id"),
                "artifact_dir": str(sample_dir),
                "prompts": prompt.artifacts,
                **sample_artifacts,
            }
        )

    workflow_doc = merge_workflow_documents(
        per_sample_docs,
        selected_payload,
        provider=cloud_config.provider,
        model=cloud_config.model,
        base_url=cloud_config.base_url,
    )
    workflow_doc.setdefault("generation_metadata", {})["prompt_builder"] = args.prompt_builder
    workflow_doc.setdefault("generation_metadata", {})["resume_reused_raw_responses"] = reused_count
    workflow_doc.setdefault("generation_metadata", {})["resume_generated_requests"] = generated_count

    workflow_path = output_dir / "workflow.json"
    write_json(workflow_path, workflow_doc, ensure_ascii=True)
    visualization_paths: list[Path] = []
    if not args.no_visualize:
        visualization_paths = wfgen_plan.render_visualizations(workflow_doc, output_dir / "visualizations")
    summary_path = wfgen_plan.write_summary(output_dir, selected_payload, workflow_doc, visualization_paths, prompt_args)
    manifest_path = wfgen_plan.write_manifest(
        output_dir,
        run_dir,
        prompt_args,
        selection,
        selected_path,
        workflow_path,
        visualization_paths,
        summary_path,
        generation_artifacts,
    )
    print(
        f"[resume-plan] complete reused={reused_count} generated={generated_count} "
        f"workflow_count={len(workflow_doc.get('workflows') or [])}",
        flush=True,
    )
    print(f"[resume-plan] wrote workflow JSON: {workflow_path}", flush=True)
    print(f"[resume-plan] wrote manifest: {manifest_path}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume wfgen_plan.py cloud per-sample generation from existing raw_response.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=280)
    parser.add_argument("--prompt-builder", choices=["full"], default="full")
    parser.add_argument("--generation-mode", choices=["cloud"], default="cloud")
    parser.add_argument("--cloud-provider", default="qwen")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_QWEN_API_KEY_ENV)
    parser.add_argument("--cloud-config", type=Path, default=wfgen_plan.DEFAULT_CLOUD_CONFIG_PATH)
    parser.add_argument("--cloud-timeout", type=int, default=300)
    parser.add_argument("--cloud-temperature", type=float, default=0.0)
    parser.add_argument("--cloud-max-tokens", type=int, default=8192)
    parser.add_argument("--extra-instruction", default=None)
    parser.add_argument("--no-insert-llm-checker-nodes", action="store_true")
    parser.add_argument("--selection-strategy", choices=["diverse", "fixed"], default="diverse")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-empty-actions", action="store_true")
    parser.add_argument("--tau-repo", default=wfgen_plan.TAU_BENCH_REPO)
    parser.add_argument("--tau-config", default=wfgen_plan.TAU_BENCH_CONFIG)
    parser.add_argument("--tau-split", default=wfgen_plan.TAU_BENCH_SPLIT)
    parser.add_argument("--no-visualize", action="store_true")
    return parser.parse_args(argv)


def prompt_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def build_cloud_config(args: argparse.Namespace) -> CloudLLMConfig:
    cloud_config_path = wfgen_plan.resolve_cloud_config_path(args.cloud_config)
    local_cloud_config = wfgen_plan.read_local_cloud_config(cloud_config_path)
    base_url = args.base_url
    model = args.model
    if local_cloud_config and args.base_url == DEFAULT_QWEN_BASE_URL:
        base_url = str(
            local_cloud_config.get("base_url")
            or local_cloud_config.get("openAiCompatible")
            or local_cloud_config.get("openai_compatible")
            or base_url
        )
    if local_cloud_config and args.model == DEFAULT_QWEN_MODEL and local_cloud_config.get("model"):
        model = str(local_cloud_config["model"])
    return CloudLLMConfig(
        provider=args.cloud_provider,
        model=model,
        base_url=base_url,
        api_key_env=args.api_key_env,
        api_key_file=str(cloud_config_path) if cloud_config_path else None,
        timeout_seconds=args.cloud_timeout,
        temperature=args.cloud_temperature,
        max_tokens=args.cloud_max_tokens,
    )


def sample_payload_for(selected_payload: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": selected_payload.get("source_dataset"),
        "config": selected_payload.get("config"),
        "split": selected_payload.get("split"),
        "selection": {
            "method": "single_sample_cloud_generation",
            "count": 1,
            "indices": [sample.get("selection_index")],
        },
        "samples": [sample],
    }


def load_existing_sample_doc(
    raw_path: Path,
    *,
    sample_payload: dict[str, Any],
    cloud_config: CloudLLMConfig,
    insert_checker_nodes: bool,
) -> dict[str, Any] | None:
    if not raw_path.exists():
        return None
    try:
        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if raw_payload.get("status") != "ok" or not raw_payload.get("content"):
        return None
    workflow_doc = parse_workflow_json(str(raw_payload["content"]))
    workflow_doc = hydrate_tool_parameters_from_samples(workflow_doc, sample_payload)
    if insert_checker_nodes:
        workflow_doc = insert_llm_checker_nodes(workflow_doc)
    workflow_doc["generation_metadata"] = {
        "generation_mode": "cloud",
        "provider": cloud_config.provider,
        "model": cloud_config.model,
        "base_url": cloud_config.base_url,
        "response_format": raw_payload.get("response_format"),
        "resumed_from_raw_response": str(raw_path),
    }
    validate_cloud_workflow(workflow_doc, sample_payload)
    return workflow_doc


if __name__ == "__main__":
    raise SystemExit(main())
