# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures.

The integration fixtures run the *real* scheduler, subprocess machinery, job store
and HTTP stack. Only the pipeline commands are swapped: instead of 3DGRUT and a
14B video model, each stage becomes a short Python program that writes the same
files at the same paths. That keeps the substituted surface to one function per
stage while everything the service itself does stays under test.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from splat_api.app import pipeline
from splat_api.app.config import Settings, hash_api_key, load_settings
from splat_api.app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]

WRITE_KEY = "test-write-key-000000000000000000"
ADMIN_KEY = "test-admin-key-111111111111111111"
READ_KEY = "test-read-key-2222222222222222222"


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A Settings instance pointed entirely at ``tmp_path``."""
    monkeypatch.setenv("SPLAT_API_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SPLAT_API_REPO_ROOT", str(REPO_ROOT))
    monkeypatch.setenv(
        "SPLAT_API_KEYS",
        f"writer:{WRITE_KEY}:read+write|ops:{ADMIN_KEY}:admin|viewer:{READ_KEY}:read",
    )
    monkeypatch.setenv("SPLAT_API_CUDA_DEVICES", "")
    monkeypatch.setenv("SPLAT_API_MAX_CONCURRENT_JOBS", "2")
    monkeypatch.setenv("SPLAT_API_STAGE_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("SPLAT_API_RECONSTRUCTION_STEPS_DEFAULT", "100")
    monkeypatch.setenv("SPLAT_API_ARTIFIXER3D_STEPS_DEFAULT", "100")
    monkeypatch.setenv("SPLAT_API_LOG_LEVEL", "WARNING")
    # Job polling in tests would otherwise exhaust the bucket; the limiter has its
    # own unit tests plus a dedicated integration test with a low limit.
    monkeypatch.setenv("SPLAT_API_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.delenv("SPLAT_API_ARTIFIXER_CHECKPOINT", raising=False)
    loaded = load_settings()
    loaded.ensure_directories()
    return loaded


@pytest.fixture
def artifixer_settings(settings: Settings, tmp_path: Path) -> Settings:
    """Settings with a stand-in ArtiFixer checkpoint so all modes are offered."""
    checkpoint = tmp_path / "artifixer-test.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    return replace(settings, artifixer_checkpoint=checkpoint, artifixer_model_id="test/model")


# --- fake pipeline ---------------------------------------------------------

_FAKE_PREPARE = r"""
import json, os, sys, struct
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
prepared = Path(args["--output_root"])
scene_id = prepared.name
steps = int(args["--reconstruction_steps"])
phases = args["--phases"].split(",")
scene_dir = Path(args["--colmap_dir"])
names = sorted(p.name for p in (scene_dir / "images").iterdir())
selected_file = args.get("--selected_image_names_file")
if selected_file:
    selected_names = [line.strip() for line in Path(selected_file).read_text().splitlines() if line.strip()]
else:
    selected_names = names
selected = [names.index(name) for name in selected_names]

trajectory_path = args.get("--trajectory_path")
frames = []
for name in names:
    frames.append({"file_path": "images/" + name, "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})
if trajectory_path:
    traj = json.loads(Path(trajectory_path).read_text())
    target_frames = [{"transform_matrix": f["transform_matrix"]} for f in traj["frames"]]
    all_frames = target_frames + [frames[i] for i in selected]
    selected_indices = list(range(len(target_frames), len(target_frames) + len(selected)))
    target_indices = list(range(len(target_frames)))
else:
    all_frames = frames
    selected_indices = selected
    target_indices = None

transforms = {"camera_model": "OPENCV", "w": 64, "h": 48, "fl_x": 60.0, "fl_y": 60.0,
              "cx": 32.0, "cy": 24.0, "frames": all_frames}
input_dir = prepared / "3dgrut_input" / scene_id
(input_dir / "nerfstudio").mkdir(parents=True, exist_ok=True)
(input_dir / "nerfstudio" / "transforms.json").write_text(json.dumps(transforms))
(input_dir / "images").mkdir(parents=True, exist_ok=True)
for name in names:
    (input_dir / "images" / name).write_bytes((scene_dir / "images" / name).read_bytes())
(input_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
for binary in ("cameras.bin", "images.bin", "points3D.bin"):
    (input_dir / "sparse" / "0" / binary).write_bytes((scene_dir / "sparse" / "0" / binary).read_bytes())
(prepared / "selected_indices.json").write_text(json.dumps(selected_indices))
(prepared / "selected_images.txt").write_text("\n".join(selected_names) + "\n")

if "reconstruct" in phases and "--reconstruction_checkpoint" not in args:
    ckpt_dir = prepared / "3dgrut_runs" / scene_id / scene_id / ("ours_%d" % steps)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, os.environ["SPLAT_API_TEST_HELPERS"])
    from fake_checkpoint import write_fake_checkpoint
    write_fake_checkpoint(ckpt_dir / ("ckpt_%d.pt" % steps), num_gaussians=64)

render_dir = prepared / "recon_results" / scene_id / "reconstruction" / scene_id / ("ours_%d" % steps)
if trajectory_path:
    render_dir = render_dir / "trajectory"
    render_count = len(target_indices)
else:
    render_count = len(all_frames)
if "render" in phases:
    for kind in ("renders", "opacity"):
        (render_dir / kind).mkdir(parents=True, exist_ok=True)
        for index in range(render_count):
            (render_dir / kind / ("%05d.png" % index)).write_bytes(b"\x89PNG\r\n\x1a\n")
    (render_dir / "selected_indices.json").write_text(json.dumps(selected_indices))
    (render_dir / "trajectory.mp4").write_bytes(b"fake-mp4")

scale = float(args.get("--metric_scale", "1.25"))
if "scale" in phases:
    (prepared / "metric_alignment").mkdir(parents=True, exist_ok=True)
    (prepared / "metric_alignment" / "scale_info.txt").write_text("Scale factor: %s\n" % scale)
if "caption" in phases:
    (prepared / "captions" / scene_id).mkdir(parents=True, exist_ok=True)
    (prepared / "captions" / scene_id / "caption.h5").write_bytes(b"fake-h5")

if "scale" in phases:
    entry = {
        "scene_id": scene_id,
        "transforms_path": "3dgrut_input/%s/nerfstudio/transforms.json" % scene_id,
        "image_root": "3dgrut_input/%s" % scene_id,
        "render_dir": str(render_dir.relative_to(prepared) / "renders"),
        "opacity_dir": str(render_dir.relative_to(prepared) / "opacity"),
        "selected_indices_path": str(render_dir.relative_to(prepared) / "selected_indices.json"),
        "prompt_path": "captions/%s/caption.h5" % scene_id,
        "reconstruction_checkpoint": args.get(
            "--reconstruction_checkpoint",
            "3dgrut_runs/%s/%s/ours_%d/ckpt_%d.pt" % (scene_id, scene_id, steps, steps),
        ),
        "metric_scale": scale,
        "camera_scale": scale * 0.01,
    }
    if target_indices is not None:
        (prepared / "trajectory").mkdir(parents=True, exist_ok=True)
        (prepared / "trajectory" / "target_indices.json").write_text(json.dumps(target_indices))
        entry["target_indices_path"] = "trajectory/target_indices.json"
        entry["has_gt"] = False
    (prepared / "split.json").write_text(json.dumps({"test": {scene_id: entry}}))

print("prepared_scene=%s" % scene_id)
print("selected_views=%d" % len(selected))
print("metric_scale=%s" % scale)
"""

_FAKE_INFERENCE = r"""
import json, sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
split_path = Path(args["--split_path"])
save_dir = Path(args["--save_dir"])
checkpoint = Path(args["--checkpoint_pt"])
render_trajectory = args["--render_trajectory"]
split = json.loads(split_path.read_text())["test"]
scene_id = next(iter(split))
entry = split[scene_id]
base = split_path.parent
transforms = json.loads((base / entry["transforms_path"]).read_text())
selected = set(json.loads((base / entry["selected_indices_path"]).read_text()))
if render_trajectory == "trajectory":
    indices = json.loads((base / entry["target_indices_path"]).read_text())
else:
    indices = [i for i in range(len(transforms["frames"]))]
run_name = (
    "distilled_views_reconstructed_colmap_auto12_evenly_spaced_sink7_"
    + ("trajectory" if render_trajectory == "trajectory" else "all_frames")
)
output_dir = save_dir / checkpoint.stem / run_name
frames_dir = output_dir / scene_id / "frames" / "batch_0000" / "pred"
frames_dir.mkdir(parents=True, exist_ok=True)
for index in indices:
    if index not in selected:
        (frames_dir / ("%05d.png" % index)).write_bytes(b"\x89PNG\r\n\x1a\n")
print("Writing outputs to %s" % output_dir)
print("Done!")
"""

_FAKE_ARTIFIXER3D = r"""
import json, os, sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
scene_root = Path(args["--scene_root"])
scene_id = args["--scene_id"]
output_root = Path(args["--output_root"])
steps = int(args["--artifixer3d_steps"])
frames_dir = Path(args["--artifixer_frames_dir"])
split_out = Path(args["--artifixer3d_plus_inference_split_path"])
split = json.loads(Path(args["--split_path"]).read_text())["test"][scene_id]
base = Path(args["--split_path"]).parent
transforms = json.loads((base / split["transforms_path"]).read_text())
selected = json.loads((base / split["selected_indices_path"]).read_text())
generated = [i for i in range(len(transforms["frames"])) if i not in set(selected)]
missing = [i for i in generated if not (frames_dir / ("%05d.png" % i)).is_file()]
if missing:
    print("missing frames: %s" % missing[:5], file=sys.stderr)
    raise SystemExit(1)

ckpt = output_root / "runs" / scene_id / scene_id / ("ours_%d" % steps) / ("ckpt_%d.pt" % steps)
ckpt.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, os.environ["SPLAT_API_TEST_HELPERS"])
from fake_checkpoint import write_fake_checkpoint
write_fake_checkpoint(ckpt, num_gaussians=96)

render_dir = output_root / "recon_results" / scene_id / "artifixer3d" / scene_id / ("ours_%d" % steps)
for kind in ("renders", "opacity"):
    (render_dir / kind).mkdir(parents=True, exist_ok=True)
    for index in range(len(transforms["frames"])):
        (render_dir / kind / ("%05d.png" % index)).write_bytes(b"\x89PNG\r\n\x1a\n")
(render_dir / "selected_indices.json").write_text(json.dumps(selected))
(render_dir / "trajectory.mp4").write_bytes(b"fake-mp4")

entry = dict(split)
entry["render_dir"] = str((render_dir / "renders").resolve())
entry["opacity_dir"] = str((render_dir / "opacity").resolve())
entry["selected_indices_path"] = str((render_dir / "selected_indices.json").resolve())
entry["transforms_path"] = str((base / split["transforms_path"]).resolve())
entry["image_root"] = str((base / split["image_root"]).resolve())
entry["prompt_path"] = str((base / split["prompt_path"]).resolve())
if "target_indices_path" in split:
    entry["target_indices_path"] = str((base / split["target_indices_path"]).resolve())
split_out.parent.mkdir(parents=True, exist_ok=True)
split_out.write_text(json.dumps({"test": {scene_id: entry}}))
print("artifixer3d_checkpoint=%s" % ckpt)
print("artifixer3d_render_dir=%s" % render_dir)
print("artifixer3d_plus_inference_split=%s" % split_out)
"""


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the four heavy stage commands with equivalent Python stand-ins."""
    helpers_dir = str(Path(__file__).resolve().parent)
    os.environ["SPLAT_API_TEST_HELPERS"] = helpers_dir

    def fake_prepare(settings, paths, **kwargs):
        argv = [sys.executable, "-c", _FAKE_PREPARE, "--colmap_dir", str(paths.scene_dir)]
        argv += ["--output_root", str(paths.prepared_root)]
        argv += ["--phases", pipeline.prepare_phases(kwargs["mode"])]
        argv += ["--reconstruction_steps", str(paths.reconstruction_steps)]
        if kwargs["has_selected_names"]:
            argv += ["--selected_image_names_file", str(paths.selected_names_file)]
        if kwargs["has_trajectory"]:
            argv += ["--trajectory_path", str(paths.trajectory_file)]
        if kwargs["metric_scale"] is not None:
            argv += ["--metric_scale", repr(float(kwargs["metric_scale"]))]
        if kwargs["cached_checkpoint"] is not None:
            argv += ["--reconstruction_checkpoint", str(kwargs["cached_checkpoint"])]
        return pipeline.StageCommand(
            name=pipeline.STAGE_PREPARE, argv=tuple(argv), description="fake prepare"
        )

    def fake_inference(name):
        def build(settings, paths, *, has_trajectory, inference_steps):
            split = (
                paths.artifixer3d_plus_split_path
                if name == pipeline.STAGE_ARTIFIXER3D_PLUS
                else paths.split_path
            )
            save_dir = (
                paths.artifixer3d_plus_save_dir
                if name == pipeline.STAGE_ARTIFIXER3D_PLUS
                else paths.artifixer_save_dir
            )
            argv = (
                sys.executable,
                "-c",
                _FAKE_INFERENCE,
                "--split_path",
                str(split),
                "--save_dir",
                str(save_dir),
                "--checkpoint_pt",
                str(settings.artifixer_checkpoint),
                "--render_trajectory",
                pipeline.render_trajectory_flag(has_trajectory),
            )
            return pipeline.StageCommand(name=name, argv=argv, description=f"fake {name}")

        return build

    def fake_af3d(settings, paths, *, frames_dir, base_checkpoint):
        argv = (
            sys.executable,
            "-c",
            _FAKE_ARTIFIXER3D,
            "--scene_root",
            str(paths.prepared_root),
            "--split_path",
            str(paths.split_path),
            "--scene_id",
            paths.scene_id,
            "--artifixer_frames_dir",
            str(frames_dir),
            "--output_root",
            str(paths.artifixer3d_root),
            "--artifixer3d_plus_inference_split_path",
            str(paths.artifixer3d_plus_split_path),
            "--artifixer3d_steps",
            str(paths.artifixer3d_steps),
        )
        return pipeline.StageCommand(
            name=pipeline.STAGE_ARTIFIXER3D, argv=argv, description="fake artifixer3d"
        )

    monkeypatch.setattr(pipeline, "prepare_command", fake_prepare)
    monkeypatch.setattr(pipeline, "artifixer_command", fake_inference(pipeline.STAGE_ARTIFIXER))
    monkeypatch.setattr(
        pipeline, "artifixer3d_plus_command", fake_inference(pipeline.STAGE_ARTIFIXER3D_PLUS)
    )
    monkeypatch.setattr(pipeline, "artifixer3d_command", fake_af3d)
    # Forward the helper path into stage subprocesses; the fake stages import it.
    monkeypatch.setattr(
        pipeline, "FORWARDED_ENV", (*pipeline.FORWARDED_ENV, "SPLAT_API_TEST_HELPERS")
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def artifixer_client(artifixer_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(artifixer_settings)) as test_client:
        yield test_client


@pytest.fixture
def rate_limited_client(settings: Settings) -> Iterator[TestClient]:
    """A client whose app allows a burst of 3 requests, for 429 coverage."""
    limited = replace(settings, rate_limit_per_minute=60, rate_limit_burst=3)
    with TestClient(create_app(limited)) as test_client:
        yield test_client


def auth(key: str = WRITE_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def upload_scene(client: TestClient, archive: Path, **params) -> dict:
    response = client.post(
        "/v1/scenes",
        content=archive.read_bytes(),
        headers={**auth(), "Content-Type": "application/zip"},
        params=params,
    )
    assert response.status_code in (200, 201), response.text
    return response.json()


def wait_for_job(client: TestClient, job_id: str, *, timeout: float = 120.0) -> dict:
    """Poll until the job reaches a terminal state."""
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}", headers=auth())
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["state"] in ("succeeded", "failed", "cancelled"):
            return payload
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s: {json.dumps(payload)}")


def digest_of(key: str) -> str:
    return hash_api_key(key)
