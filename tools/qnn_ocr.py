"""QNN-capable OCR tool wrapper.

The default mode is simulated so the tool can be used in CI and workflow tests.
Set `run_real_qnn: true` in the tool config to call the on-device pyagent QNN
OCR pipeline.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .abi import fail, ok


def run_ocr_extract_text(request: dict[str, Any]) -> dict[str, Any]:
    started_at = time.time()
    params = request.get("parameters") or {}
    cfg = request.get("tool_config") or {}
    image_path = params.get("image_path") or params.get("image") or cfg.get("image_path")
    mock_text = params.get("mock_text") or cfg.get("mock_text") or "QNN OCR placeholder text"

    if not cfg.get("run_real_qnn", False):
        return ok(
            artifacts={
                "ocr_text": {
                    "text": mock_text,
                    "detections": [],
                    "image_path": image_path,
                    "backend": "simulated_qnn_ocr",
                }
            },
            raw={"entrypoint": "ocr_extract_text", "run_real_qnn": False},
            started_at=started_at,
        )

    if not image_path:
        return fail("ocr_extract_text requires image_path when run_real_qnn is true", started_at=started_at)
    try:
        result = run_pyagent_qnn_ocr(image_path=str(image_path), cfg=cfg)
        return ok(artifacts={"ocr_text": result}, raw={"entrypoint": "ocr_extract_text"}, started_at=started_at)
    except Exception as exc:
        return fail(str(exc), raw={"entrypoint": "ocr_extract_text", "image_path": image_path}, started_at=started_at)


def run_pyagent_qnn_ocr(*, image_path: str, cfg: dict[str, Any]) -> dict[str, Any]:
    from pyagent.llm_utils import qnn_ocr_pipeline as pipeline_mod

    detector_cfg = cfg.get("detector") or {}
    recognizer_cfg = cfg.get("recognizer") or {}
    qnn_cfg = cfg.get("qnn") or cfg

    pipeline_mod.DETECTOR_CONFIG.update(
        {
            key: value
            for key, value in {
                "dlc_path": detector_cfg.get("dlc_path"),
                "input_list": detector_cfg.get("input_list"),
                "output_dir": detector_cfg.get("output_dir"),
            }.items()
            if value
        }
    )
    pipeline_mod.RECOGNIZER_CONFIG.update(
        {
            key: value
            for key, value in {
                "dlc_path": recognizer_cfg.get("dlc_path"),
                "input_list": recognizer_cfg.get("input_list"),
                "output_dir": recognizer_cfg.get("output_dir"),
            }.items()
            if value
        }
    )
    pipeline_mod.QNN_CONFIG.update(
        {
            key: value
            for key, value in {
                "qnn_dir": qnn_cfg.get("qnn_dir"),
                "backend": qnn_cfg.get("backend"),
                "log_level": qnn_cfg.get("log_level"),
            }.items()
            if value
        }
    )

    adb_path = qnn_cfg.get("adb") or "adb"
    adsp_library_path = qnn_cfg.get(
        "adsp_library_path",
        f"{pipeline_mod.QNN_CONFIG['qnn_dir']}/hexagon-v75/unsigned;/vendor/lib/rfsa/adsp;/dsp",
    )
    extra_args = qnn_cfg.get("extra_args") or []
    timeout = int(qnn_cfg.get("timeout", 120))

    def run_qnn_htp_adapter(**kwargs: Any) -> int:
        return run_qnn_net_run(
            adb_path=adb_path,
            device_serial=kwargs.get("device_serial"),
            qnn_dir=kwargs.get("qnn_dir"),
            dlc_path=kwargs.get("dlc_path"),
            input_list=kwargs.get("input_list"),
            output_dir=kwargs.get("output_dir"),
            backend=kwargs.get("backend", "libQnnHtp.so"),
            log_level=kwargs.get("log_level", "warn"),
            adsp_library_path=adsp_library_path,
            timeout=int(kwargs.get("timeout", timeout)),
            extra_args=extra_args,
        )

    pipeline_mod.run_qnn_htp = run_qnn_htp_adapter
    pipeline = pipeline_mod.QnnOcrPipeline(
        device_serial=qnn_cfg.get("device"),
        adb_path=adb_path,
    )
    results = pipeline.run_ocr_on_image(
        image_path,
        output_image_path=cfg.get("annotated_output"),
        detector_only=bool(cfg.get("detector_only", False)),
    )
    detections = [
        {
            "box": {"xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3]},
            "text": text,
            "confidence": float(confidence),
        }
        for box, text, confidence in results
    ]
    return {
        "text": "\n".join(item["text"] for item in detections),
        "detections": detections,
        "image_path": image_path,
        "backend": "qnn_ocr_pyagent_pipeline",
    }


def run_qnn_net_run(
    *,
    adb_path: str,
    device_serial: str | None,
    qnn_dir: str,
    dlc_path: str,
    input_list: str,
    output_dir: str,
    backend: str,
    log_level: str,
    adsp_library_path: str,
    timeout: int,
    extra_args: list[str],
) -> int:
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
    command = [adb_path]
    if device_serial:
        command += ["-s", device_serial]
    command += ["shell", "sh", "-c", remote_cmd]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": command,
                },
                ensure_ascii=True,
            )
        )
    return result.returncode


def write_example_request(path: str | Path) -> None:
    example = {
        "entrypoint": "ocr_extract_text",
        "parameters": {"image_path": "pyagent/test_image.jpg"},
        "tool_config": {
            "run_real_qnn": False,
            "mock_text": "sample OCR output",
        },
    }
    Path(path).write_text(json.dumps(example, indent=2, ensure_ascii=True), encoding="utf-8")

