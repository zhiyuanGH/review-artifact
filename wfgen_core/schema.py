"""JSON schema and prompts for LLM-generated workflow specifications."""

from __future__ import annotations

WORKFLOW_SCHEMA_NAME = "wfgen_workflow_spec"

WORKFLOW_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "schema_version": {
            "type": "string",
            "const": "wfgen.v1",
            "description": "Workflow schema version. Use wfgen.v1.",
        },
        "benchmark": {
            "$ref": "#/$defs/benchmark",
        },
        "workflows": {
            "type": "array",
            "description": "Generated workflows for the supplied benchmark tasks.",
            "items": {
                "$ref": "#/$defs/workflow",
            },
        },
        "generation_notes": {
            "type": "array",
            "description": "Important assumptions or limitations in this generated workflow set.",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "schema_version",
        "benchmark",
        "workflows",
        "generation_notes",
    ],
    "additionalProperties": False,
    "$defs": {
        "benchmark": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short benchmark name inferred from the dataset.",
                },
                "description": {
                    "type": "string",
                    "description": "One paragraph summary of the benchmark purpose.",
                },
                "source_dataset": {
                    "type": "string",
                    "description": "Input dataset path or name.",
                },
                "task_count": {
                    "type": "integer",
                    "description": "Number of tasks represented in the generated output.",
                },
                "assumptions": {
                    "type": "array",
                    "description": "Assumptions made while interpreting the dataset.",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": [
                "name",
                "description",
                "source_dataset",
                "task_count",
                "assumptions",
            ],
            "additionalProperties": False,
        },
        "workflow": {
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Stable identifier such as workflow_001.",
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable workflow name.",
                },
                "description": {
                    "type": "string",
                    "description": "Concise explanation of what the workflow accomplishes.",
                },
                "source_task_ids": {
                    "type": "array",
                    "description": "Dataset task ids or row identifiers used to create this workflow.",
                    "items": {
                        "type": "string",
                    },
                },
                "user_goal": {
                    "type": "string",
                    "description": "The user-facing goal solved by the workflow.",
                },
                "inputs": {
                    "type": "array",
                    "description": "External inputs required before the workflow starts.",
                    "items": {
                        "$ref": "#/$defs/artifact",
                    },
                },
                "nodes": {
                    "type": "array",
                    "description": "DAG nodes. Each node is one executable computation or action.",
                    "items": {
                        "$ref": "#/$defs/node",
                    },
                },
                "edges": {
                    "type": "array",
                    "description": "Typed data dependencies between nodes.",
                    "items": {
                        "$ref": "#/$defs/edge",
                    },
                },
                "outputs": {
                    "type": "array",
                    "description": "Final artifacts emitted by the workflow.",
                    "items": {
                        "$ref": "#/$defs/artifact",
                    },
                },
                "execution_hints": {
                    "$ref": "#/$defs/execution_hints",
                },
                "validation_checks": {
                    "type": "array",
                    "description": "Checks an interpreter or evaluator should run before execution.",
                    "items": {
                        "$ref": "#/$defs/validation_check",
                    },
                },
            },
            "required": [
                "workflow_id",
                "name",
                "description",
                "source_task_ids",
                "user_goal",
                "inputs",
                "nodes",
                "edges",
                "outputs",
                "execution_hints",
                "validation_checks",
            ],
            "additionalProperties": False,
        },
        "node": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Stable node id unique inside the workflow.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["llm", "app", "dnn", "control", "data"],
                    "description": "Node category. Use llm for language-model reasoning and app for external tool calls.",
                },
                "label": {
                    "type": "string",
                    "description": "Short action label.",
                },
                "purpose": {
                    "type": "string",
                    "description": "Why this node is needed.",
                },
                "entrypoint": {
                    "type": "string",
                    "description": "Model name, tool name, ABI entry, or interpreter hook.",
                },
                "parameters": {
                    "type": "array",
                    "description": "Structured node parameters as name-value pairs.",
                    "items": {
                        "$ref": "#/$defs/parameter",
                    },
                },
                "inputs": {
                    "type": "array",
                    "description": "Artifacts consumed by this node.",
                    "items": {
                        "$ref": "#/$defs/artifact",
                    },
                },
                "outputs": {
                    "type": "array",
                    "description": "Artifacts produced by this node.",
                    "items": {
                        "$ref": "#/$defs/artifact",
                    },
                },
                "preferred_backend": {
                    "type": "string",
                    "enum": ["cpu", "gpu", "npu", "cloud", "any", "unknown"],
                    "description": "Initial placement hint for future offloading decisions.",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Suggested micro-batch size. Use 1 when no batching is useful.",
                },
                "side_effects": {
                    "type": "array",
                    "description": "External side effects such as sending an email or updating a calendar.",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": [
                "node_id",
                "kind",
                "label",
                "purpose",
                "entrypoint",
                "parameters",
                "inputs",
                "outputs",
                "preferred_backend",
                "batch_size",
                "side_effects",
            ],
            "additionalProperties": False,
        },
        "edge": {
            "type": "object",
            "properties": {
                "from_node": {
                    "type": "string",
                    "description": "Producer node id.",
                },
                "to_node": {
                    "type": "string",
                    "description": "Consumer node id.",
                },
                "artifact": {
                    "$ref": "#/$defs/artifact",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this dependency is required.",
                },
            },
            "required": [
                "from_node",
                "to_node",
                "artifact",
                "reason",
            ],
            "additionalProperties": False,
        },
        "artifact": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Artifact name.",
                },
                "artifact_type": {
                    "type": "string",
                    "description": "Logical type such as Text, Image, Audio, JSON, Intent, Embedding, or Table.",
                },
                "description": {
                    "type": "string",
                    "description": "Meaning and expected contents.",
                },
                "codec": {
                    "type": "string",
                    "description": "Serialization codec such as utf8, json, jpeg, wav, protobuf, or none.",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Expected maximum serialized size in bytes. Use 0 if unknown.",
                },
            },
            "required": [
                "name",
                "artifact_type",
                "description",
                "codec",
                "max_bytes",
            ],
            "additionalProperties": False,
        },
        "parameter": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Parameter name.",
                },
                "value": {
                    "type": "string",
                    "description": "Parameter value serialized as a string.",
                },
                "description": {
                    "type": "string",
                    "description": "What this parameter controls.",
                },
            },
            "required": [
                "name",
                "value",
                "description",
            ],
            "additionalProperties": False,
        },
        "execution_hints": {
            "type": "object",
            "properties": {
                "parallel_waves": {
                    "type": "array",
                    "description": "Dependency-safe execution waves. Nodes inside a wave can run concurrently.",
                    "items": {
                        "$ref": "#/$defs/parallel_wave",
                    },
                },
                "critical_path": {
                    "type": "array",
                    "description": "Likely critical path as node ids.",
                    "items": {
                        "type": "string",
                    },
                },
                "offloading_rationale": {
                    "type": "string",
                    "description": "Why the initial backend hints were chosen.",
                },
            },
            "required": [
                "parallel_waves",
                "critical_path",
                "offloading_rationale",
            ],
            "additionalProperties": False,
        },
        "parallel_wave": {
            "type": "object",
            "properties": {
                "wave_index": {
                    "type": "integer",
                    "description": "Zero-based wave index.",
                },
                "node_ids": {
                    "type": "array",
                    "description": "Node ids that can execute together.",
                    "items": {
                        "type": "string",
                    },
                },
                "reason": {
                    "type": "string",
                    "description": "Why these nodes are dependency-safe to group.",
                },
            },
            "required": [
                "wave_index",
                "node_ids",
                "reason",
            ],
            "additionalProperties": False,
        },
        "validation_check": {
            "type": "object",
            "properties": {
                "check_id": {
                    "type": "string",
                    "description": "Stable check id.",
                },
                "description": {
                    "type": "string",
                    "description": "Validation condition.",
                },
                "expected_signal": {
                    "type": "string",
                    "description": "Observable signal indicating success.",
                },
            },
            "required": [
                "check_id",
                "description",
                "expected_signal",
            ],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You generate ODABench-style workflow specifications from benchmark datasets.

A workflow is a directed acyclic graph of executable nodes and typed edges.
Use nodes for LLM inference, app or tool invocation, DNN inference, control flow, or data preparation.
Use edges only for real data dependencies. If two nodes do not depend on each other, leave them unconnected so the interpreter can run them in parallel.
Prefer explicit artifacts over vague dependencies. Include inputs, outputs, codecs, and approximate max_bytes when possible.
Return only JSON that conforms to the provided schema.
"""


def response_text_format() -> dict:
    """Return the Responses API structured-output format for this schema."""

    return {
        "format": {
            "type": "json_schema",
            "name": WORKFLOW_SCHEMA_NAME,
            "strict": True,
            "schema": WORKFLOW_SCHEMA,
        }
    }
