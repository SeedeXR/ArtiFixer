#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end acceptance test against a real GPU and a real COLMAP scene.

Nothing is stubbed: this starts the service as its own process, talks to it over
HTTP, and runs the actual 3DGRUT / ArtiFixer pipeline. It is the check that the
service produces a genuine Gaussian splat, not just that its plumbing is
internally consistent (that is what ``splat_api/tests`` covers).

    python -m splat_api.tools.e2e_real \
        --colmap-scene /data/colmap-scenes/truck \
        --mode reconstruct \
        --reconstruction-steps 1000

For the correction path, point it at a release checkpoint:

    python -m splat_api.tools.e2e_real \
        --colmap-scene /data/colmap-scenes/truck \
        --mode artifixer3d \
        --checkpoint /data/artifixer-checkpoints/artifixer-1.3b.pt \
        --reconstruction-steps 1000 --artifixer3d-steps 1000 \
        --selected-views 24 --metric-scale 1.0

Exit code 0 means a splat was produced and parsed successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[e2e {time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_archive(scene_dir: Path, archive_path: Path) -> int:
    """Zip a COLMAP scene the way a client would."""
    scene_dir = scene_dir.resolve()
    images = sorted(
        path
        for path in (scene_dir / "images").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No images found under {scene_dir / 'images'}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as bundle:
        for image in images:
            bundle.write(image, arcname=f"images/{image.name}")
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            bundle.write(scene_dir / "sparse" / "0" / name, arcname=f"sparse/0/{name}")
    log(f"archive: {archive_path} ({archive_path.stat().st_size / 1e6:.1f} MB, {len(images)} images)")
    return len(images)


class Client:
    """Minimal HTTP client over urllib, so the test adds no dependencies."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: float = 120.0,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(f"{self._base}{path}", data=body, method=method)
        request.add_header("Authorization", f"Bearer {self._key}")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except urllib.error.URLError as error:
            # The server is still binding its socket. Status 0 means "no HTTP
            # response yet" so callers can retry rather than crash during boot.
            return 0, str(error.reason).encode()

    def json(self, method: str, path: str, payload: Any = None, **kwargs) -> tuple[int, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        status, raw = self.request(
            method, path, body=body, content_type="application/json" if body else None, **kwargs
        )
        try:
            return status, json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return status, raw.decode("utf-8", errors="replace")


def start_server(data_root: Path, api_key: str, port: int, checkpoint: Path | None) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "SPLAT_API_DATA_ROOT": str(data_root),
            "SPLAT_API_REPO_ROOT": str(REPO_ROOT),
            "SPLAT_API_KEYS": f"e2e:{api_key}:admin",
            "SPLAT_API_PORT": str(port),
            "SPLAT_API_HOST": "127.0.0.1",
            "SPLAT_API_LOG_LEVEL": "INFO",
            "SPLAT_API_RATE_LIMIT_PER_MINUTE": "0",
            "SPLAT_API_MAX_CONCURRENT_JOBS": "1",
            "SPLAT_API_RECONSTRUCTION_STEPS_MAX": "200000",
            "PYTHONPATH": str(REPO_ROOT),
            # Without this the child's log lines sit in a pipe buffer and are lost
            # if it is signalled, which makes a boot failure undiagnosable.
            "PYTHONUNBUFFERED": "1",
        }
    )
    if checkpoint is not None:
        env["SPLAT_API_ARTIFIXER_CHECKPOINT"] = str(checkpoint)
        env["SPLAT_API_ARTIFIXER_MODEL_ID"] = (
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            if "1.3b" in checkpoint.name.lower()
            else "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        )
        log(f"checkpoint: {checkpoint.name} -> {env['SPLAT_API_ARTIFIXER_MODEL_ID']}")
    else:
        env.pop("SPLAT_API_ARTIFIXER_CHECKPOINT", None)

    log_path = data_root / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "splat_api.app.main"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"server pid {process.pid}, log {log_path}")
    return process


def wait_ready(client: Client, process: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited early with code {process.returncode}")
        status, payload = client.json("GET", "/readyz", timeout=10.0)
        if status == 200:
            log(f"ready: {json.dumps(payload['checks'])}")
            return
        time.sleep(1.0)
    raise SystemExit("server did not become ready")


def poll_job(client: Client, job_id: str, *, timeout: float, stages: list[str]) -> dict:
    deadline = time.monotonic() + timeout
    last_state: tuple[str, float] | None = None
    while time.monotonic() < deadline:
        status, job = client.json("GET", f"/v1/jobs/{job_id}")
        if status != 200:
            raise SystemExit(f"job poll failed: {status} {job}")
        signature = (job["state"], job["progress"])
        if signature != last_state:
            running = [stage["name"] for stage in job["stages"] if stage["state"] == "running"]
            log(f"state={job['state']} progress={job['progress']:.2f} running={running or '-'}")
            last_state = signature
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(5.0)
    raise SystemExit(f"job {job_id} did not finish within {timeout}s")


def verify_ply(path: Path) -> dict[str, Any]:
    """Parse the delivered splat and sanity-check its contents."""
    from plyfile import PlyData

    data = PlyData.read(str(path))
    element = data.elements[0]
    names = {prop.name for prop in element.properties}
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "f_dc_0"}
    missing = required - names
    if missing:
        raise SystemExit(f"PLY is missing 3DGS properties: {sorted(missing)}")
    if len(element.data) == 0:
        raise SystemExit("PLY contains zero Gaussians")
    import numpy as np

    positions = np.stack([element.data["x"], element.data["y"], element.data["z"]], axis=1)
    if not np.isfinite(positions).all():
        raise SystemExit("PLY contains non-finite positions")
    return {
        "num_gaussians": int(len(element.data)),
        "num_properties": len(names),
        "sh_rest_terms": sum(1 for name in names if name.startswith("f_rest_")),
        "bbox_min": positions.min(axis=0).round(3).tolist(),
        "bbox_max": positions.max(axis=0).round(3).tolist(),
        "bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-scene", type=Path, required=True)
    parser.add_argument("--mode", choices=("reconstruct", "artifixer3d", "artifixer3d_plus"), default="reconstruct")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reconstruction-steps", type=int, default=1000)
    parser.add_argument("--artifixer3d-steps", type=int, default=1000)
    parser.add_argument(
        "--selected-views",
        type=int,
        default=0,
        help="Use only the first N images as 3DGRUT anchors (0 = all). Required > 0 for artifixer3d.",
    )
    parser.add_argument("--metric-scale", type=float, default=None)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to a temp directory.")
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args(argv)

    if args.mode != "reconstruct" and args.checkpoint is None:
        parser.error(f"--checkpoint is required for --mode {args.mode}")

    temporary_root = args.data_root is None
    data_root = args.data_root or Path(tempfile.mkdtemp(prefix="splat-api-e2e-"))
    data_root.mkdir(parents=True, exist_ok=True)
    api_key = "e2e-" + secrets.token_hex(16)
    client = Client(f"http://127.0.0.1:{args.port}", api_key)
    process = start_server(data_root, api_key, args.port, args.checkpoint)
    started = time.monotonic()

    try:
        wait_ready(client, process)

        status, capabilities = client.json("GET", "/v1/capabilities")
        if status != 200:
            raise SystemExit(f"capabilities failed: {status} {capabilities}")
        log(f"modes={capabilities['modes']} gpus={capabilities['gpu_count']}")
        if args.mode not in capabilities["modes"]:
            raise SystemExit(f"mode {args.mode} is not offered by this deployment")

        archive_path = data_root / "scene.zip"
        image_count = build_archive(args.colmap_scene, archive_path)

        upload_started = time.monotonic()
        status, scene = client.request(
            "POST",
            "/v1/scenes?label=e2e",
            body=archive_path.read_bytes(),
            content_type="application/zip",
            timeout=1800.0,
        )
        if status not in (200, 201):
            raise SystemExit(f"upload failed: {status} {scene.decode(errors='replace')}")
        scene = json.loads(scene)
        log(
            f"scene {scene['scene_id']}: {scene['image_count']} images, "
            f"{scene['point_count']} points, {scene['colmap_width']}x{scene['colmap_height']}, "
            f"validated in {time.monotonic() - upload_started:.1f}s"
        )
        if scene["image_count"] != image_count:
            raise SystemExit("server image count does not match the archive")

        payload: dict[str, Any] = {
            "scene_id": scene["scene_id"],
            "mode": args.mode,
            "reconstruction_steps": args.reconstruction_steps,
            "artifixer3d_steps": args.artifixer3d_steps,
            "inference_steps": args.inference_steps,
        }
        if args.selected_views > 0:
            payload["selected_image_names"] = scene["image_names"][: args.selected_views]
        if args.metric_scale is not None:
            payload["metric_scale"] = args.metric_scale

        status, job = client.json("POST", "/v1/jobs", payload)
        if status != 202:
            raise SystemExit(f"job creation failed: {status} {json.dumps(job)}")
        job_id = job["job_id"]
        log(f"job {job_id} queued: stages={[stage['name'] for stage in job['stages']]}")

        job = poll_job(client, job_id, timeout=args.timeout, stages=[s["name"] for s in job["stages"]])
        for stage in job["stages"]:
            log(
                f"  stage {stage['name']:<18} {stage['state']:<10} "
                f"{stage['duration_seconds'] or 0:.1f}s"
            )
        if job["state"] != "succeeded":
            for stage in job["stages"]:
                if stage["state"] == "failed":
                    _, tail = client.request("GET", f"/v1/jobs/{job_id}/logs/{stage['name']}")
                    print(tail.decode("utf-8", errors="replace")[-4000:], file=sys.stderr)
            raise SystemExit(f"job {job_id} finished as {job['state']}: {job['error']}")

        artifacts = {artifact["name"]: artifact for artifact in job["artifacts"]}
        log(f"artifacts: {sorted(artifacts)}")
        if "splat.ply" not in artifacts:
            raise SystemExit("job succeeded but produced no splat.ply")

        status, blob = client.request("GET", f"/v1/jobs/{job_id}/artifacts/splat.ply", timeout=1800.0)
        if status != 200:
            raise SystemExit(f"splat download failed: {status}")
        splat_path = data_root / "splat.ply"
        splat_path.write_bytes(blob)

        import hashlib

        digest = hashlib.sha256(blob).hexdigest()
        if artifacts["splat.ply"]["sha256"] != digest:
            raise SystemExit("downloaded splat does not match its advertised digest")

        summary = verify_ply(splat_path)
        log(f"splat verified: {json.dumps(summary)}")
        log(f"total wall clock: {time.monotonic() - started:.1f}s")
        print(json.dumps({"result": "ok", "job": job_id, "mode": args.mode, "splat": summary}, indent=2))
        return 0
    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        if temporary_root and not args.keep_data:
            shutil.rmtree(data_root, ignore_errors=True)
        else:
            log(f"data root kept at {data_root}")


if __name__ == "__main__":
    raise SystemExit(main())
