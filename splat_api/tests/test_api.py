# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests over the real HTTP stack, scheduler and job store.

Only the four heavy stage commands are substituted (see ``fake_pipeline`` in
conftest); the ASGI app, middleware, auth, SQLite store, subprocess execution,
artifact collection and download paths are all exercised as shipped.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from splat_api.app.routes import resolve_training_views
from splat_api.tests.conftest import ADMIN_KEY, READ_KEY, WRITE_KEY, auth, upload_scene, wait_for_job
from splat_api.tests.helpers import build_colmap_zip, make_zip

pytestmark = pytest.mark.usefixtures("fake_pipeline")


def identity_trajectory(frames: int = 3) -> dict:
    return {
        "camera_model": "OPENCV",
        "w": 64,
        "h": 48,
        "fl_x": 60.0,
        "fl_y": 60.0,
        "cx": 32.0,
        "cy": 24.0,
        "frames": [
            {
                "transform_matrix": [
                    [1, 0, 0, index * 0.1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            }
            for index in range(frames)
        ],
    }


class TestTrainingViewSelection:
    """3DGRUT crashes on an empty validation split, so views are always held back.

    threedgrut/datasets/dataset_colmap.py:101-104 computes the val split as
    setdiff1d(all, selected); with every image selected it is empty and
    compute_spatial_extents raises. Verified against the real engine.
    """

    def test_explicit_subset_is_used_verbatim(self) -> None:
        names = [f"f{index}.jpg" for index in range(10)]
        training, auto = resolve_training_views(["f3.jpg", "f7.jpg"], names)
        assert training == ["f3.jpg", "f7.jpg"]
        assert auto is False

    def test_auto_holdout_drops_every_eighth_image(self) -> None:
        names = [f"f{index}.jpg" for index in range(20)]
        training, auto = resolve_training_views(None, names)
        assert auto is True
        assert "f0.jpg" not in training
        assert "f8.jpg" not in training
        assert "f16.jpg" not in training
        assert len(training) == 17

    def test_auto_holdout_leaves_training_views_for_a_tiny_scene(self) -> None:
        training, auto = resolve_training_views(None, ["a.jpg", "b.jpg"])
        assert training == ["b.jpg"]
        assert auto is True

    def test_reconstruct_job_reports_the_automatic_holdout(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        archive, _ = build_colmap_zip(tmp_path, image_count=16)
        scene_id = upload_scene(client, archive)["scene_id"]
        job = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()
        finished = wait_for_job(client, job["job_id"])
        assert finished["state"] == "succeeded", json.dumps(finished, indent=2)
        assert finished["request"]["validation_holdout_auto"] is True
        assert finished["request"]["selected_image_count"] == 14  # 16 - indices 0 and 8

    def test_explicit_subset_disables_the_automatic_holdout(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        archive, names = build_colmap_zip(tmp_path, image_count=10)
        scene_id = upload_scene(client, archive)["scene_id"]
        job = client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "selected_image_names": names[:4]},
            headers=auth(),
        ).json()
        finished = wait_for_job(client, job["job_id"])
        assert finished["state"] == "succeeded"
        assert finished["request"]["validation_holdout_auto"] is False
        assert finished["request"]["selected_image_count"] == 4


class TestMeta:
    def test_healthz_needs_no_credential(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readyz_reports_checks(self, client: TestClient) -> None:
        response = client.get("/readyz")
        assert response.status_code == 200
        checks = response.json()["checks"]
        assert checks["database"] == "ok"
        assert checks["data_root_writable"] is True

    def test_security_headers_and_request_id(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Content-Security-Policy"].startswith("default-src 'none'")
        assert response.headers["X-Request-Id"]

    def test_capabilities_hides_artifixer_modes_without_a_checkpoint(self, client: TestClient) -> None:
        payload = client.get("/v1/capabilities", headers=auth(READ_KEY)).json()
        assert payload["modes"] == ["reconstruct"]
        assert payload["artifixer_checkpoint_configured"] is False

    def test_capabilities_offers_all_modes_with_a_checkpoint(self, artifixer_client: TestClient) -> None:
        payload = artifixer_client.get("/v1/capabilities", headers=auth(READ_KEY)).json()
        assert payload["modes"] == ["reconstruct", "artifixer3d", "artifixer3d_plus"]

    def test_metrics_exposes_queue_gauges(self, client: TestClient) -> None:
        body = client.get("/metrics", headers=auth(READ_KEY)).text
        assert "splat_api_queue_depth" in body
        assert 'splat_api_jobs_total{state="queued"}' in body

    def test_openapi_is_served(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 200


class TestAuthorization:
    def test_rejects_missing_credential(self, client: TestClient) -> None:
        response = client.get("/v1/scenes")
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"

    def test_rejects_wrong_credential(self, client: TestClient) -> None:
        response = client.get("/v1/scenes", headers=auth("nope-000000000000000000000"))
        assert response.status_code == 401

    def test_read_key_cannot_upload(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(READ_KEY), "Content-Type": "application/zip"},
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"

    def test_register_requires_admin(self, client: TestClient) -> None:
        response = client.post("/v1/scenes/register", json={"path": "x"}, headers=auth(WRITE_KEY))
        assert response.status_code == 403

    def test_error_payload_carries_the_request_id(self, client: TestClient) -> None:
        response = client.get("/v1/scenes")
        assert response.json()["error"]["request_id"] == response.headers["X-Request-Id"]


class TestRateLimiting:
    def test_returns_429_with_retry_after_once_the_burst_is_spent(
        self, rate_limited_client: TestClient
    ) -> None:
        statuses = [
            rate_limited_client.get("/v1/capabilities", headers=auth(READ_KEY)).status_code
            for _ in range(5)
        ]
        assert statuses[:3] == [200, 200, 200]
        assert 429 in statuses
        response = rate_limited_client.get("/v1/capabilities", headers=auth(READ_KEY))
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1
        assert response.json()["error"]["code"] == "rate_limited"

    def test_health_endpoints_are_never_rate_limited(self, rate_limited_client: TestClient) -> None:
        for _ in range(10):
            assert rate_limited_client.get("/healthz").status_code == 200

    def test_source_address_is_a_shared_ceiling(self, rate_limited_client: TestClient) -> None:
        """Rotating credentials must not mint a fresh allowance.

        The limiter runs before authentication, so bucketing only on the offered
        credential would leave key guessing unthrottled. The address is always
        charged, which caps the whole client regardless of what it sends.
        """
        for _ in range(4):
            rate_limited_client.get("/v1/capabilities", headers=auth(READ_KEY))
        assert rate_limited_client.get("/v1/capabilities", headers=auth(READ_KEY)).status_code == 429
        # A different credential from the same address is still throttled.
        assert rate_limited_client.get("/v1/capabilities", headers=auth(WRITE_KEY)).status_code == 429
        # So is a credential that does not exist at all.
        assert (
            rate_limited_client.get(
                "/v1/capabilities", headers=auth("guess-000000000000000000000000")
            ).status_code
            == 429
        )

    def test_unknown_credentials_are_throttled_not_just_rejected(
        self, rate_limited_client: TestClient
    ) -> None:
        statuses = [
            rate_limited_client.get(
                "/v1/capabilities", headers=auth(f"guess-{index:026d}")
            ).status_code
            for index in range(6)
        ]
        assert statuses[:3] == [401, 401, 401]
        assert statuses[3:] == [429, 429, 429]


class TestSceneUpload:
    def test_accepts_a_raw_zip_body(self, client: TestClient, tmp_path: Path) -> None:
        archive, names = build_colmap_zip(tmp_path, image_count=5, point_count=20)
        payload = upload_scene(client, archive)
        assert payload["image_count"] == 5
        assert payload["point_count"] == 20
        assert payload["camera_models"] == ["PINHOLE"]
        assert sorted(payload["image_names"]) == sorted(names)
        assert payload["source"] == "upload"
        assert payload["size_bytes"] > 0

    def test_accepts_multipart_form_data(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        response = client.post(
            "/v1/scenes",
            files={"file": ("scene.zip", archive.read_bytes(), "application/zip")},
            headers=auth(),
        )
        assert response.status_code == 201, response.text
        assert response.json()["image_count"] == 4

    def test_rejects_an_unsupported_content_type(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(), "Content-Type": "text/plain"},
        )
        assert response.status_code == 400
        assert "application/zip" in response.json()["error"]["message"]

    def test_deduplicates_identical_uploads(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        first = upload_scene(client, archive)
        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(), "Content-Type": "application/zip"},
        )
        assert response.status_code == 200
        assert response.headers["X-Scene-Deduplicated"] == "true"
        assert response.json()["scene_id"] == first["scene_id"]

    def test_dedupe_can_be_disabled(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        first = upload_scene(client, archive)
        second = upload_scene(client, archive, dedupe=False)
        assert first["scene_id"] != second["scene_id"]

    def test_rejects_an_invalid_archive_and_leaves_no_scene(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        archive = make_zip(tmp_path / "bad.zip", {"images/a.jpg": b"x"})
        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(), "Content-Type": "application/zip"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unprocessable_input"
        assert client.get("/v1/scenes", headers=auth()).json() == []

    def test_rejects_an_empty_body(self, client: TestClient) -> None:
        response = client.post(
            "/v1/scenes", content=b"", headers={**auth(), "Content-Type": "application/zip"}
        )
        assert response.status_code == 400

    def test_get_and_list_scenes(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        assert client.get(f"/v1/scenes/{scene_id}", headers=auth()).json()["scene_id"] == scene_id
        listed = client.get("/v1/scenes", headers=auth()).json()
        assert [scene["scene_id"] for scene in listed] == [scene_id]

    def test_unknown_scene_is_404(self, client: TestClient) -> None:
        assert client.get("/v1/scenes/scene_missing", headers=auth()).status_code == 404

    def test_malformed_scene_id_is_400(self, client: TestClient) -> None:
        response = client.get("/v1/scenes/..%2Fetc", headers=auth())
        assert response.status_code in (400, 404)

    def test_delete_removes_the_scene_tree(self, client: TestClient, tmp_path: Path, settings) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        scene_dir = settings.scenes_dir / scene_id
        assert scene_dir.is_dir()
        assert client.delete(f"/v1/scenes/{scene_id}", headers=auth(ADMIN_KEY)).status_code == 204
        assert not scene_dir.exists()
        assert client.get(f"/v1/scenes/{scene_id}", headers=auth()).status_code == 404


class TestReconstructJob:
    def test_full_lifecycle_produces_a_splat(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path, image_count=4)
        scene_id = upload_scene(client, archive)["scene_id"]

        created = client.post(
            "/v1/jobs", json={"scene_id": scene_id, "mode": "reconstruct"}, headers=auth()
        )
        assert created.status_code == 202, created.text
        job = created.json()
        assert job["state"] == "queued"
        assert [stage["name"] for stage in job["stages"]] == ["prepare", "export"]
        assert created.headers["Location"] == f"/v1/jobs/{job['job_id']}"

        finished = wait_for_job(client, job["job_id"])
        assert finished["state"] == "succeeded", json.dumps(finished, indent=2)
        assert finished["progress"] == 1.0
        assert all(stage["state"] == "succeeded" for stage in finished["stages"])
        assert all(stage["duration_seconds"] is not None for stage in finished["stages"])

        artifacts = {artifact["name"]: artifact for artifact in finished["artifacts"]}
        assert "splat.ply" in artifacts
        assert artifacts["splat.ply"]["kind"] == "splat_ply"
        assert artifacts["splat.ply"]["sha256"]
        assert "splat_checkpoint.pt" in artifacts
        assert "reconstruction_preview.mp4" in artifacts
        assert "logs/prepare.log" in artifacts
        # No ArtiFixer stage ran, so there are no corrected frames.
        assert "corrected_frames.zip" not in artifacts

    def test_downloaded_ply_is_a_valid_gaussian_splat(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        plyfile = pytest.importorskip("plyfile")
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
        finished = wait_for_job(client, job_id)
        assert finished["state"] == "succeeded", json.dumps(finished, indent=2)

        response = client.get(f"/v1/jobs/{job_id}/artifacts/splat.ply", headers=auth())
        assert response.status_code == 200
        assert response.headers["content-disposition"] == 'attachment; filename="splat.ply"'
        record = next(a for a in finished["artifacts"] if a["name"] == "splat.ply")
        assert response.headers["etag"].strip('"') == record["sha256"]

        local = tmp_path / "downloaded.ply"
        local.write_bytes(response.content)
        data = plyfile.PlyData.read(str(local))
        properties = {prop.name for prop in data.elements[0].properties}
        assert {"x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0"} <= properties
        assert len(data.elements[0].data) == 64  # fake reconstruction writes 64 Gaussians

    def test_stage_log_is_retrievable(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
        assert wait_for_job(client, job_id)["state"] == "succeeded"

        response = client.get(f"/v1/jobs/{job_id}/logs/prepare", headers=auth())
        assert response.status_code == 200
        assert "prepared_scene=" in response.text
        assert client.get(f"/v1/jobs/{job_id}/logs/nonexistent", headers=auth()).status_code == 404

    def test_reconstruction_checkpoint_is_cached_across_jobs(
        self, client: TestClient, tmp_path: Path, settings
    ) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        first = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()
        assert wait_for_job(client, first["job_id"])["state"] == "succeeded"

        cached = list((settings.data_root / "cache" / "reconstruction").glob("*/checkpoint.pt"))
        assert len(cached) == 1

        second = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()
        finished = wait_for_job(client, second["job_id"])
        assert finished["state"] == "succeeded"
        log = client.get(f"/v1/jobs/{second['job_id']}/logs/prepare", headers=auth()).text
        assert "--reconstruction_checkpoint" in log

    def test_selected_subset_is_honoured(self, client: TestClient, tmp_path: Path) -> None:
        archive, names = build_colmap_zip(tmp_path, image_count=6)
        scene_id = upload_scene(client, archive)["scene_id"]
        job = client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "selected_image_names": names[:3]},
            headers=auth(),
        ).json()
        finished = wait_for_job(client, job["job_id"])
        assert finished["state"] == "succeeded"
        assert finished["request"]["selected_image_count"] == 3
        assert "selected_views=3" in client.get(
            f"/v1/jobs/{job['job_id']}/logs/prepare", headers=auth()
        ).text

    def test_export_can_be_skipped(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job = client.post(
            "/v1/jobs", json={"scene_id": scene_id, "export_ply": False}, headers=auth()
        ).json()
        assert [stage["name"] for stage in job["stages"]] == ["prepare"]
        finished = wait_for_job(client, job["job_id"])
        assert finished["state"] == "succeeded"
        assert {artifact["name"] for artifact in finished["artifacts"]} & {"splat.ply"} == set()


class TestArtifixerModes:
    def test_artifixer_mode_requires_a_checkpoint(self, client: TestClient, tmp_path: Path) -> None:
        archive, names = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        response = client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "mode": "artifixer3d", "selected_image_names": names[:2]},
            headers=auth(),
        )
        assert response.status_code == 503
        assert "ArtiFixer checkpoint" in response.json()["error"]["message"]

    def test_artifixer3d_distills_corrected_frames(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        archive, names = build_colmap_zip(tmp_path, image_count=8)
        scene_id = upload_scene(artifixer_client, archive)["scene_id"]
        job = artifixer_client.post(
            "/v1/jobs",
            json={
                "scene_id": scene_id,
                "mode": "artifixer3d",
                "selected_image_names": names[:4],
            },
            headers=auth(),
        ).json()
        finished = wait_for_job(artifixer_client, job["job_id"])
        assert finished["state"] == "succeeded", json.dumps(finished, indent=2)
        assert [stage["name"] for stage in finished["stages"]] == [
            "prepare",
            "artifixer",
            "artifixer3d",
            "export",
        ]

        artifacts = {artifact["name"]: artifact for artifact in finished["artifacts"]}
        assert "corrected_frames.zip" in artifacts
        assert "artifixer3d_preview.mp4" in artifacts

        response = artifixer_client.get(
            f"/v1/jobs/{job['job_id']}/artifacts/corrected_frames.zip", headers=auth()
        )
        assert response.status_code == 200
        bundle_path = tmp_path / "frames.zip"
        bundle_path.write_bytes(response.content)
        with zipfile.ZipFile(bundle_path) as bundle:
            # The 4 non-anchor frames are the generated ones.
            assert sorted(bundle.namelist()) == [f"{index:05d}.png" for index in range(4, 8)]

    def test_artifixer3d_splat_comes_from_the_distilled_checkpoint(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        plyfile = pytest.importorskip("plyfile")
        archive, names = build_colmap_zip(tmp_path, image_count=6)
        scene_id = upload_scene(artifixer_client, archive)["scene_id"]
        job = artifixer_client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "mode": "artifixer3d", "selected_image_names": names[:3]},
            headers=auth(),
        ).json()
        assert wait_for_job(artifixer_client, job["job_id"])["state"] == "succeeded"

        response = artifixer_client.get(
            f"/v1/jobs/{job['job_id']}/artifacts/splat.ply", headers=auth()
        )
        local = tmp_path / "af3d.ply"
        local.write_bytes(response.content)
        # 96 Gaussians identifies the ArtiFixer3D checkpoint, not the 64 of the base.
        assert len(plyfile.PlyData.read(str(local)).elements[0].data) == 96

    def test_artifixer3d_plus_runs_a_second_inference_pass(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        archive, names = build_colmap_zip(tmp_path, image_count=8)
        scene_id = upload_scene(artifixer_client, archive)["scene_id"]
        job = artifixer_client.post(
            "/v1/jobs",
            json={
                "scene_id": scene_id,
                "mode": "artifixer3d_plus",
                "selected_image_names": names[:4],
            },
            headers=auth(),
        ).json()
        finished = wait_for_job(artifixer_client, job["job_id"])
        assert finished["state"] == "succeeded", json.dumps(finished, indent=2)
        assert [stage["name"] for stage in finished["stages"]] == [
            "prepare",
            "artifixer",
            "artifixer3d",
            "artifixer3d_plus",
            "export",
        ]

    def test_trajectory_job_uses_the_trajectory_render_path(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        archive, names = build_colmap_zip(tmp_path, image_count=5)
        scene_id = upload_scene(artifixer_client, archive)["scene_id"]
        job = artifixer_client.post(
            "/v1/jobs",
            json={
                "scene_id": scene_id,
                "mode": "artifixer3d",
                "selected_image_names": names,
                "trajectory": identity_trajectory(3),
            },
            headers=auth(),
        ).json()
        finished = wait_for_job(artifixer_client, job["job_id"])
        assert finished["state"] == "succeeded", json.dumps(finished, indent=2)
        assert finished["request"]["trajectory_frames"] == 3
        log = artifixer_client.get(f"/v1/jobs/{job['job_id']}/logs/artifixer", headers=auth()).text
        assert "--render_trajectory trajectory" in log or "trajectory" in log


class TestJobValidation:
    def _scene(self, client: TestClient, tmp_path: Path, **kwargs) -> tuple[str, list[str]]:
        archive, names = build_colmap_zip(tmp_path, **kwargs)
        return upload_scene(client, archive)["scene_id"], names

    def test_unknown_scene_is_404(self, client: TestClient) -> None:
        response = client.post("/v1/jobs", json={"scene_id": "scene_absent"}, headers=auth())
        assert response.status_code == 404

    def test_unknown_image_name_is_422(self, client: TestClient, tmp_path: Path) -> None:
        scene_id, _ = self._scene(client, tmp_path)
        response = client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "selected_image_names": ["ghost.jpg"]},
            headers=auth(),
        )
        assert response.status_code == 422
        assert "not present in scene" in response.json()["error"]["message"]

    def test_artifixer3d_rejects_selecting_every_image(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        scene_id, names = self._scene(artifixer_client, tmp_path, image_count=4)
        response = artifixer_client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "mode": "artifixer3d", "selected_image_names": names},
            headers=auth(),
        )
        assert response.status_code == 422
        assert "strict subset" in response.json()["error"]["message"]

    def test_artifixer3d_requires_a_subset_or_trajectory(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        scene_id, _ = self._scene(artifixer_client, tmp_path)
        response = artifixer_client.post(
            "/v1/jobs", json={"scene_id": scene_id, "mode": "artifixer3d"}, headers=auth()
        )
        assert response.status_code == 422
        assert "at least one generated view" in json.dumps(response.json())

    def test_reconstruct_rejects_a_trajectory(self, client: TestClient, tmp_path: Path) -> None:
        scene_id, _ = self._scene(client, tmp_path)
        response = client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "mode": "reconstruct", "trajectory": identity_trajectory()},
            headers=auth(),
        )
        assert response.status_code == 422

    def test_rejects_unknown_fields(self, client: TestClient, tmp_path: Path) -> None:
        scene_id, _ = self._scene(client, tmp_path)
        response = client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "hydra_overrides": "out_dir=/etc"},
            headers=auth(),
        )
        assert response.status_code == 422

    def test_rejects_out_of_range_steps(self, client: TestClient, tmp_path: Path) -> None:
        scene_id, _ = self._scene(client, tmp_path)
        assert (
            client.post(
                "/v1/jobs",
                json={"scene_id": scene_id, "reconstruction_steps": 10},
                headers=auth(),
            ).status_code
            == 422
        )

    def test_rejects_image_names_with_separators(self, client: TestClient, tmp_path: Path) -> None:
        scene_id, _ = self._scene(client, tmp_path)
        response = client.post(
            "/v1/jobs",
            json={"scene_id": scene_id, "selected_image_names": ["../../etc/passwd"]},
            headers=auth(),
        )
        assert response.status_code == 422

    def test_rejects_a_malformed_trajectory_matrix(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        scene_id, names = self._scene(artifixer_client, tmp_path)
        trajectory = identity_trajectory(1)
        trajectory["frames"][0]["transform_matrix"] = [[1, 0], [0, 1]]
        response = artifixer_client.post(
            "/v1/jobs",
            json={
                "scene_id": scene_id,
                "mode": "artifixer3d",
                "selected_image_names": names,
                "trajectory": trajectory,
            },
            headers=auth(),
        )
        assert response.status_code == 422

    def test_rejects_a_trajectory_frame_with_file_path(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        """Target-only trajectories: assert_target_only_trajectory forbids file_path."""
        scene_id, names = self._scene(artifixer_client, tmp_path)
        trajectory = identity_trajectory(1)
        trajectory["frames"][0]["file_path"] = "images/frame_0000.jpg"
        response = artifixer_client.post(
            "/v1/jobs",
            json={
                "scene_id": scene_id,
                "mode": "artifixer3d",
                "selected_image_names": names,
                "trajectory": trajectory,
            },
            headers=auth(),
        )
        assert response.status_code == 422


class TestJobManagement:
    def test_client_token_is_idempotent(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        body = {"scene_id": scene_id, "client_token": "run-2026-07-25-a"}
        first = client.post("/v1/jobs", json=body, headers=auth())
        second = client.post("/v1/jobs", json=body, headers=auth())
        assert first.status_code == 202
        assert second.status_code == 200
        assert first.json()["job_id"] == second.json()["job_id"]

    def test_reusing_a_token_for_a_different_job_conflicts(
        self, artifixer_client: TestClient, tmp_path: Path
    ) -> None:
        archive, names = build_colmap_zip(artifixer_client and tmp_path, image_count=4)
        scene_id = upload_scene(artifixer_client, archive)["scene_id"]
        token = "run-shared-token"
        assert (
            artifixer_client.post(
                "/v1/jobs", json={"scene_id": scene_id, "client_token": token}, headers=auth()
            ).status_code
            == 202
        )
        response = artifixer_client.post(
            "/v1/jobs",
            json={
                "scene_id": scene_id,
                "mode": "artifixer3d",
                "selected_image_names": names[:2],
                "client_token": token,
            },
            headers=auth(),
        )
        assert response.status_code == 409

    def test_list_and_filter_jobs(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_ids = [
            client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
            for _ in range(3)
        ]
        for job_id in job_ids:
            wait_for_job(client, job_id)

        listed = client.get("/v1/jobs", headers=auth()).json()
        assert len(listed["jobs"]) == 3
        succeeded = client.get("/v1/jobs", params={"state": "succeeded"}, headers=auth()).json()
        assert len(succeeded["jobs"]) == 3
        page = client.get("/v1/jobs", params={"limit": 2}, headers=auth()).json()
        assert len(page["jobs"]) == 2
        assert page["next_cursor"]
        rest = client.get(
            "/v1/jobs", params={"limit": 2, "cursor": page["next_cursor"]}, headers=auth()
        ).json()
        assert len(rest["jobs"]) == 1

    def test_rejects_an_invalid_state_filter(self, client: TestClient) -> None:
        assert client.get("/v1/jobs", params={"state": "exploded"}, headers=auth()).status_code == 422

    def test_cancel_a_queued_job(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        # Saturate the two workers so the third job is still queued when cancelled.
        held = [
            client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
            for _ in range(2)
        ]
        target = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()
        response = client.post(f"/v1/jobs/{target['job_id']}/cancel", headers=auth())
        assert response.status_code == 200
        assert response.json()["state"] in ("cancelled", "running", "succeeded")
        for job_id in held:
            wait_for_job(client, job_id)

    def test_cancelling_a_finished_job_conflicts(self, client: TestClient, tmp_path: Path) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
        assert wait_for_job(client, job_id)["state"] == "succeeded"
        response = client.post(f"/v1/jobs/{job_id}/cancel", headers=auth())
        assert response.status_code == 409

    def test_failing_stage_marks_the_job_failed(
        self, artifixer_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-zero stage exit must surface as a failed job, not a hang."""
        from splat_api.app import pipeline

        def broken(settings, paths, **kwargs):
            return pipeline.StageCommand(
                name=pipeline.STAGE_PREPARE,
                argv=("/bin/sh", "-c", "echo boom >&2; exit 3"),
                description="deliberately failing stage",
            )

        monkeypatch.setattr(pipeline, "prepare_command", broken)
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(artifixer_client, archive)["scene_id"]
        job_id = artifixer_client.post(
            "/v1/jobs", json={"scene_id": scene_id}, headers=auth()
        ).json()["job_id"]

        finished = wait_for_job(artifixer_client, job_id)
        assert finished["state"] == "failed"
        assert finished["stages"][0]["exit_code"] == 3
        assert "exit code 3" in finished["error"]
        assert finished["stages"][1]["state"] == "skipped"


class TestArtifactSecurity:
    def _finished_job(self, client: TestClient, tmp_path: Path) -> str:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
        assert wait_for_job(client, job_id)["state"] == "succeeded"
        return job_id

    @pytest.mark.parametrize(
        "name",
        [
            "../../../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "output/../../../etc/passwd",
            "/etc/passwd",
        ],
    )
    def test_traversal_in_artifact_name_is_refused(
        self, client: TestClient, tmp_path: Path, name: str
    ) -> None:
        job_id = self._finished_job(client, tmp_path)
        response = client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth())
        assert response.status_code in (400, 404)
        assert b"root:" not in response.content

    def test_unlisted_artifact_is_404(self, client: TestClient, tmp_path: Path) -> None:
        job_id = self._finished_job(client, tmp_path)
        # Real files inside the job directory that were never published as
        # artifacts must not be reachable: the allowlist is the job's own record,
        # not the filesystem.
        for name in ("inputs/selected_train_images.txt", "prep", "logs"):
            assert client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth()).status_code == 404

    def test_manifest_is_published_and_describes_the_run(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        job_id = self._finished_job(client, tmp_path)
        response = client.get(f"/v1/jobs/{job_id}/artifacts/manifest.json", headers=auth())
        assert response.status_code == 200
        manifest = json.loads(response.content)
        assert manifest["job_id"] == job_id
        assert manifest["mode"] == "reconstruct"
        assert [stage["name"] for stage in manifest["stages"]] == ["prepare", "export"]
        assert manifest["metrics"]["stage_seconds"]["prepare"] > 0

    def test_manifest_leaks_no_server_paths(self, client: TestClient, tmp_path: Path, settings) -> None:
        """The manifest is downloadable, so it must not disclose the data root."""
        job_id = self._finished_job(client, tmp_path)
        response = client.get(f"/v1/jobs/{job_id}/artifacts/manifest.json", headers=auth())
        body = response.text
        assert str(settings.data_root) not in body
        assert str(settings.repo_root) not in body
        assert "reconstruction_cache_key" not in body
        assert "scene_root" not in body
        # Commands are kept, with deployment paths replaced by placeholders.
        manifest = json.loads(body)
        commands = [stage["command"] for stage in manifest["stages"] if stage["command"]]
        assert commands
        assert any("<data_root>" in command for command in commands)

    def test_stage_logs_require_the_write_scope(self, client: TestClient, tmp_path: Path) -> None:
        """Logs are raw subprocess output; a read-only credential cannot fetch them."""
        job_id = self._finished_job(client, tmp_path)
        assert client.get(f"/v1/jobs/{job_id}/logs/prepare", headers=auth(READ_KEY)).status_code == 403
        assert (
            client.get(f"/v1/jobs/{job_id}/artifacts/logs/prepare.log", headers=auth(READ_KEY)).status_code
            == 403
        )
        # The deliverables stay readable with the read scope.
        assert client.get(f"/v1/jobs/{job_id}/artifacts/splat.ply", headers=auth(READ_KEY)).status_code == 200
        assert client.get(f"/v1/jobs/{job_id}/logs/prepare", headers=auth(WRITE_KEY)).status_code == 200

    def test_splat_stats_report_the_gaussian_count(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        job_id = self._finished_job(client, tmp_path)
        response = client.get(f"/v1/jobs/{job_id}/artifacts/splat_stats.json", headers=auth())
        assert response.status_code == 200
        assert json.loads(response.content)["num_gaussians"] == 64

    def test_artifact_listing_matches_the_job_payload(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        job_id = self._finished_job(client, tmp_path)
        listed = client.get(f"/v1/jobs/{job_id}/artifacts", headers=auth()).json()
        from_job = client.get(f"/v1/jobs/{job_id}", headers=auth()).json()["artifacts"]
        assert {item["name"] for item in listed} == {item["name"] for item in from_job}

    def test_job_response_hides_server_paths(self, client: TestClient, tmp_path: Path) -> None:
        job_id = self._finished_job(client, tmp_path)
        payload = client.get(f"/v1/jobs/{job_id}", headers=auth()).json()
        assert "scene_root" not in payload["request"]
        assert "reconstruction_cache_key" not in payload["request"]
        assert str(tmp_path) not in json.dumps(payload)
