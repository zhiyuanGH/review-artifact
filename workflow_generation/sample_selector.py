"""Tau-bench sample selection for workflow generation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..workflow_planning.generate import (
    TAU_BENCH_CONFIG,
    TAU_BENCH_REPO,
    TAU_BENCH_SPLIT,
    normalize_tau_bench_task,
)


CATEGORY_PRIORITY = [
    "retail_cancel_order",
    "retail_return_items",
    "retail_exchange_multi_item",
    "retail_modify_multi_item",
    "retail_payment_update",
    "retail_user_address_update",
    "retail_compound_address_item_update",
    "airline_cancel_reservation",
    "airline_update_flights",
    "airline_update_baggages",
    "retail_order_address_update",
    "retail_exchange_single_item",
    "retail_modify_single_item",
    "retail_compound_cancel_return",
]


def select_tau_bench_samples(
    *,
    sample_count: int,
    selection_strategy: str = "diverse",
    sample_offset: int = 0,
    seed: int = 0,
    include_empty_actions: bool = False,
    tau_repo: str = TAU_BENCH_REPO,
    tau_config: str = TAU_BENCH_CONFIG,
    tau_split: str = TAU_BENCH_SPLIT,
    selected_samples: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if selected_samples:
        samples, selection = load_selected_samples(selected_samples, sample_count)
    else:
        samples, selection = load_and_select_from_hf(
            sample_count=sample_count,
            selection_strategy=selection_strategy,
            sample_offset=sample_offset,
            seed=seed,
            include_empty_actions=include_empty_actions,
            tau_repo=tau_repo,
            tau_config=tau_config,
            tau_split=tau_split,
        )
    payload = {
        "source_dataset": tau_repo,
        "config": tau_config,
        "split": tau_split,
        "selection": selection,
        "samples": samples,
    }
    return payload, selection


def load_selected_samples(path: Path, sample_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if not isinstance(samples, list):
        raise ValueError(f"--selected-samples does not contain a samples list: {path}")
    samples = [sample for sample in samples if isinstance(sample, dict)]
    if not samples:
        raise ValueError(f"--selected-samples contains no samples: {path}")
    if len(samples) > sample_count:
        samples = samples[:sample_count]
    selection = dict(payload.get("selection") or {}) if isinstance(payload, dict) else {}
    selection.update(
        {
            "method": selection.get("method", "existing_selected_samples"),
            "count": len(samples),
            "indices": [sample.get("selection_index") for sample in samples],
            "source_file": str(path),
        }
    )
    return samples, selection


def load_and_select_from_hf(
    *,
    sample_count: int,
    selection_strategy: str,
    sample_offset: int,
    seed: int,
    include_empty_actions: bool,
    tau_repo: str,
    tau_config: str,
    tau_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: datasets. Install it in the active Python environment.") from exc

    dataset = load_dataset(tau_repo, name=tau_config, split=tau_split)
    normalized = [
        normalize_tau_bench_task(row, selection_index=idx)
        for idx, row in enumerate(dataset)
    ]

    if selection_strategy == "fixed":
        end = sample_offset + sample_count
        if sample_offset < 0 or end > len(normalized):
            raise ValueError(f"Fixed window {sample_offset}:{end} is outside dataset size {len(normalized)}")
        samples = [annotate_sample(sample, "fixed_window") for sample in normalized[sample_offset:end]]
        return samples, {
            "method": "fixed_contiguous_window",
            "offset": sample_offset,
            "count": len(samples),
            "indices": [sample["selection_index"] for sample in samples],
        }

    samples = select_diverse(normalized, sample_count, seed=seed, include_empty=include_empty_actions)
    return samples, {
        "method": "auto_diverse",
        "count": len(samples),
        "indices": [sample["selection_index"] for sample in samples],
        "seed": seed,
        "criteria": [
            "maximize domain and action-sequence coverage",
            "prefer executable samples with ground-truth tool actions",
            "choose deterministic row positions for reproducibility",
        ],
    }


def select_diverse(
    samples: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    include_empty: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates = []
    for sample in samples:
        actions = sample.get("ground_truth_actions") or []
        if not actions and not include_empty:
            continue
        category = category_for_sample(sample)
        annotated = annotate_sample(sample, category)
        grouped[category].append(annotated)
        candidates.append(annotated)

    for group_samples in grouped.values():
        group_samples.sort(key=lambda item: item["selection_index"])
        if seed:
            rng.shuffle(group_samples)

    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for category in CATEGORY_PRIORITY:
        if len(selected) >= count:
            break
        if grouped.get(category):
            add_if_new(selected, seen_indices, grouped[category][0])

    remaining_categories = sorted(
        (category for category in grouped if category not in CATEGORY_PRIORITY),
        key=lambda category: (grouped[category][0].get("domain", ""), category),
    )
    for category in remaining_categories:
        if len(selected) >= count:
            break
        add_if_new(selected, seen_indices, grouped[category][0])

    candidates.sort(key=lambda item: item["selection_index"])
    if seed:
        rng.shuffle(candidates)
    for sample in candidates:
        if len(selected) >= count:
            break
        add_if_new(selected, seen_indices, sample)

    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} candidate samples available for requested count {count}")
    return selected


def add_if_new(selected: list[dict[str, Any]], seen_indices: set[int], sample: dict[str, Any]) -> None:
    idx = int(sample["selection_index"])
    if idx in seen_indices:
        return
    seen_indices.add(idx)
    selected.append(sample)


def category_for_sample(sample: dict[str, Any]) -> str:
    domain = sample.get("domain") or "unknown"
    actions = [action.get("name", "") for action in sample.get("ground_truth_actions") or []]
    action_set = set(actions)
    product_reads = actions.count("get_product_details")

    if domain == "airline":
        if "update_reservation_flights" in action_set:
            return "airline_update_flights"
        if "update_reservation_baggages" in action_set:
            return "airline_update_baggages"
        if "cancel_order" in action_set:
            return "airline_cancel_reservation"
        return "airline_other"

    if "cancel_order" in action_set and "return_delivered_order_items" in action_set:
        return "retail_compound_cancel_return"
    if "modify_pending_order_address" in action_set and "modify_pending_order_items" in action_set:
        return "retail_compound_address_item_update"
    if "exchange_delivered_order_items" in action_set:
        return "retail_exchange_multi_item" if product_reads >= 2 else "retail_exchange_single_item"
    if "modify_pending_order_items" in action_set:
        return "retail_modify_multi_item" if product_reads >= 2 else "retail_modify_single_item"
    if "return_delivered_order_items" in action_set:
        return "retail_return_items"
    if "cancel_order" in action_set:
        return "retail_cancel_order"
    if "modify_pending_order_payment" in action_set:
        return "retail_payment_update"
    if "modify_pending_order_address" in action_set:
        return "retail_order_address_update"
    if "modify_user_address" in action_set:
        return "retail_user_address_update"
    if not actions:
        return f"{domain}_no_actions"
    return f"{domain}_other"


def annotate_sample(sample: dict[str, Any], category: str) -> dict[str, Any]:
    result = json.loads(json.dumps(sample, ensure_ascii=False))
    actions = [action.get("name", "") for action in result.get("ground_truth_actions") or []]
    result["selection_reason"] = category.replace("_", " ")
    result["diversity_tags"] = [
        str(result.get("domain") or "unknown"),
        category,
        "actions:" + "->".join(actions),
    ]
    return result
