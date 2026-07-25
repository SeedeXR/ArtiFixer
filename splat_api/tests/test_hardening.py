# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for specific defects found by adversarial review.

Each test here corresponds to a concrete failure that was reproduced against an
earlier revision of this service. They exist so those failures cannot come back
quietly; the docstrings record what went wrong.
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from splat_api.app import pipeline
from splat_api.app.config import Settings
from splat_api.app.errors import BadRequest, UnprocessableInput
from splat_api.app.jobstore import JobRecord, JobStore, StageRecord, utc_now
from splat_api.tests.conftest import READ_KEY, auth, upload_scene, wait_for_job
from splat_api.tests.helpers import build_colmap_scene, build_colmap_zip, make_zip


class TestStageOutputCannotWedgeAWorker:
    """A stage emitting a huge unbroken run of output used to deadlock the worker.

    ``StreamReader.readline`` raises ``ValueError`` past the 64 KiB stream limit.
    That killed the drain task, so nobody read the pipe; the child then blocked on
    write and never exited, and because a paused stdout transport never reports
    EOF, ``process.wait()`` could never complete — even after SIGKILL. The job sat
    in ``running`` until the 24-hour stage timeout, the worker was permanently
    lost, and both cancel and shutdown hung on it.
    """

    @pytest.mark.usefixtures("fake_pipeline")
    def test_carriage_return_progress_bar_does_not_hang(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        program = (
            "import sys\n"
            # 4000 x ~64 chars of CR-separated progress, then a real line: over
            # 250 KB with no newline until the very end.
            "for index in range(4000):\n"
            "    sys.stdout.write('\\rstep %d/4000 loss=0.123456 elapsed=00:00:00' % index)\n"
            "sys.stdout.write('\\nprepared_scene=done\\n')\n"
        )

        def noisy(settings, paths, **kwargs):
            return pipeline.StageCommand(
                name=pipeline.STAGE_PREPARE,
                argv=(sys.executable, "-c", program),
                description="stage that emits a very long unbroken line",
            )

        monkeypatch.setattr(pipeline, "prepare_command", noisy)
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post(
            "/v1/jobs", json={"scene_id": scene_id, "export_ply": False}, headers=auth()
        ).json()["job_id"]

        # The point is that this returns at all, and quickly.
        started = time.monotonic()
        finished = wait_for_job(client, job_id, timeout=90.0)
        assert time.monotonic() - started < 90.0
        assert finished["state"] == "succeeded", json.dumps(finished, indent=2)
        assert finished["stages"][0]["exit_code"] == 0

    @pytest.mark.usefixtures("fake_pipeline")
    def test_the_log_still_captures_the_output(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        program = "import sys; sys.stdout.write('x' * 300000); sys.stdout.write('\\ndone\\n')"

        def noisy(settings, paths, **kwargs):
            return pipeline.StageCommand(
                name=pipeline.STAGE_PREPARE, argv=(sys.executable, "-c", program), description="noisy"
            )

        monkeypatch.setattr(pipeline, "prepare_command", noisy)
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post(
            "/v1/jobs", json={"scene_id": scene_id, "export_ply": False}, headers=auth()
        ).json()["job_id"]
        assert wait_for_job(client, job_id, timeout=90.0)["state"] == "succeeded"

        log = client.get(f"/v1/jobs/{job_id}/logs/prepare", headers=auth(), params={"tail_bytes": 262144}).text
        assert "done" in log


@pytest.mark.usefixtures("fake_pipeline")
class TestNoLeakedStagingDirectories:
    """A deduplicated upload used to leave its staging directory and full ZIP behind.

    The dedupe hit returned from inside the ``try``, and cleanup lived only in the
    ``except`` path, so a caller could fill the disk by re-POSTing one archive.
    """

    def test_dedupe_hit_leaves_no_staging_directory(
        self, client: TestClient, tmp_path: Path, settings: Settings
    ) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        upload_scene(client, archive)
        for _ in range(3):
            upload_scene(client, archive)
        assert list(settings.uploads_dir.iterdir()) == []

    def test_rejected_upload_leaves_no_staging_directory(
        self, client: TestClient, tmp_path: Path, settings: Settings
    ) -> None:
        archive = make_zip(tmp_path / "bad.zip", {"images/a.jpg": b"x"})
        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(), "Content-Type": "application/zip"},
        )
        assert response.status_code == 422
        assert list(settings.uploads_dir.iterdir()) == []

    def test_successful_upload_leaves_no_staging_directory(
        self, client: TestClient, tmp_path: Path, settings: Settings
    ) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        assert list(settings.uploads_dir.iterdir()) == []
        assert (settings.scenes_dir / scene_id).is_dir()


@pytest.mark.usefixtures("fake_pipeline")
class TestQueueAdmissionFailure:
    """A full queue used to leave a phantom `queued` row with no queue slot.

    Such a job never ran, never reached a terminal state, consumed its
    ``client_token``, and made its scene permanently undeletable because the
    "no active jobs" check counted it.
    """

    @pytest.fixture
    def tiny_queue_client(self, settings: Settings, monkeypatch: pytest.MonkeyPatch):
        from dataclasses import replace

        from splat_api.app.main import create_app

        # One queue slot and one worker: the third submission must be refused.
        limited = replace(settings, queue_capacity=1, max_concurrent_jobs=1)
        with TestClient(create_app(limited)) as client:
            yield client

    def test_rejected_submissions_leave_no_rows(
        self, tiny_queue_client: TestClient, tmp_path: Path
    ) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(tiny_queue_client, archive)["scene_id"]
        statuses = [
            tiny_queue_client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).status_code
            for _ in range(8)
        ]
        assert 503 in statuses, statuses
        accepted = statuses.count(202)

        listed = tiny_queue_client.get("/v1/jobs", params={"limit": 200}, headers=auth()).json()["jobs"]
        assert len(listed) == accepted

        for job in listed:
            wait_for_job(tiny_queue_client, job["job_id"])
        # With no phantom rows the scene is deletable once real jobs finish.
        from splat_api.tests.conftest import ADMIN_KEY

        assert (
            tiny_queue_client.delete(f"/v1/scenes/{scene_id}", headers=auth(ADMIN_KEY)).status_code == 204
        )

    def test_a_refused_token_can_be_retried(
        self, tiny_queue_client: TestClient, tmp_path: Path
    ) -> None:
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(tiny_queue_client, archive)["scene_id"]
        body = {"scene_id": scene_id, "client_token": "retry-me-please"}
        statuses = []
        for _ in range(8):
            statuses.append(tiny_queue_client.post("/v1/jobs", json=body, headers=auth()).status_code)
        # Either it was accepted, or every attempt was refused; a refused attempt
        # must not have burned the token on a dead row.
        assert set(statuses) <= {202, 200, 503}
        if 503 in statuses and 202 not in statuses:
            for job in tiny_queue_client.get("/v1/jobs", headers=auth()).json()["jobs"]:
                wait_for_job(tiny_queue_client, job["job_id"])
            assert tiny_queue_client.post("/v1/jobs", json=body, headers=auth()).status_code == 202


@pytest.mark.usefixtures("fake_pipeline")
class TestBadInputIsNotAServerError:
    """Three caller-supplied archives used to produce 500s with tracebacks."""

    def test_corrupt_member_data(self, client: TestClient, tmp_path: Path) -> None:
        scene = tmp_path / "scene"
        build_colmap_scene(scene)
        archive = tmp_path / "crc.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(scene.rglob("*")):
                if path.is_file():
                    bundle.write(path, arcname=path.relative_to(scene).as_posix())
        # Corrupt stored image bytes without touching the CRC in either header.
        raw = bytearray(archive.read_bytes())
        marker = raw.find(b"\xff\xd8")  # JPEG SOI of the first stored image
        if marker < 0:
            marker = raw.find(b"\x89PNG")
        assert marker > 0
        raw[marker + 8 : marker + 24] = b"\x00" * 16
        archive.write_bytes(bytes(raw))

        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(), "Content-Type": "application/zip"},
        )
        assert response.status_code in (400, 422), response.text
        assert response.json()["error"]["code"] != "internal_error"

    def test_unsupported_compression_method(self, client: TestClient, tmp_path: Path) -> None:
        archive = make_zip(tmp_path / "ppmd.zip", {"images/a.jpg": b"x" * 512})
        raw = bytearray(archive.read_bytes())
        # Method field: +8 in a local header, +10 in a central directory record.
        for signature, offset in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
            position = raw.find(signature)
            while position >= 0:
                raw[position + offset : position + offset + 2] = (98).to_bytes(2, "little")
                position = raw.find(signature, position + 4)
        archive.write_bytes(bytes(raw))

        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(), "Content-Type": "application/zip"},
        )
        assert response.status_code == 400
        assert "compression" in response.json()["error"]["message"]

    def test_undecodable_image_bytes(self, client: TestClient, tmp_path: Path) -> None:
        scene = tmp_path / "scene"
        names = build_colmap_scene(scene)
        (scene / "images" / names[0]).write_bytes(b"this is definitely not a JPEG")
        from splat_api.tests.helpers import zip_directory

        archive = zip_directory(scene, tmp_path / "scene.zip")
        response = client.post(
            "/v1/scenes",
            content=archive.read_bytes(),
            headers={**auth(), "Content-Type": "application/zip"},
        )
        assert response.status_code == 422
        assert "not a decodable image" in response.json()["error"]["message"]

    def test_colmap_image_name_with_a_newline(self, tmp_path: Path) -> None:
        """A name with a newline would corrupt the selected-views file."""
        from splat_api.app.colmap_input import validate_scene
        from splat_api.tests.helpers import FakeImage, write_images_bin

        build_colmap_scene(tmp_path, image_count=2)
        write_images_bin(
            tmp_path / "sparse" / "0" / "images.bin",
            [
                FakeImage(image_id=1, name="frame_0000.jpg"),
                FakeImage(image_id=2, name="frame\n0001.jpg"),
            ],
        )
        with pytest.raises(UnprocessableInput, match="characters this service does not accept"):
            validate_scene(tmp_path, max_images=100)


@pytest.mark.usefixtures("fake_pipeline")
class TestUploadLimits:
    def test_multipart_without_content_length_is_refused(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Starlette spools multipart parts to /tmp with no cap of its own.

        Without a declared length there is nothing to check before that happens, so
        such a request is refused outright rather than allowed to fill the disk.
        """
        archive, _ = build_colmap_zip(tmp_path)
        payload = archive.read_bytes()
        boundary = "----splatapitestboundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="scene.zip"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()

        def chunks():
            yield body

        response = client.post(
            "/v1/scenes",
            content=chunks(),
            headers={
                **auth(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Transfer-Encoding": "chunked",
            },
        )
        assert response.status_code == 400
        assert "Content-Length" in response.json()["error"]["message"]

    def test_oversized_json_body_is_refused_before_parsing(self, client: TestClient) -> None:
        from dataclasses import replace

        from splat_api.app.main import create_app

        small = replace(client.app.state.settings, max_json_bytes=2048)
        with TestClient(create_app(small)) as limited:
            response = limited.post(
                "/v1/jobs",
                json={"scene_id": "scene_x", "client_token": "a" * 4096},
                headers=auth(),
            )
            assert response.status_code == 413

    def test_trajectory_frame_count_is_bounded_at_parse_time(self) -> None:
        """Unbounded frames cost seconds of event-loop time before any check."""
        from splat_api.app.schemas import MAX_TRAJECTORY_FRAMES_PARSE, JobCreate

        frame = {"transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        with pytest.raises(Exception) as excinfo:
            JobCreate(
                scene_id="scene_x",
                mode="reconstruct",
                trajectory={
                    "camera_model": "OPENCV",
                    "w": 64,
                    "h": 48,
                    "fl_x": 60.0,
                    "fl_y": 60.0,
                    "cx": 32.0,
                    "cy": 24.0,
                    "frames": [frame] * (MAX_TRAJECTORY_FRAMES_PARSE + 1),
                },
            )
        assert "too_long" in str(excinfo.value) or "at most" in str(excinfo.value)


class TestStoreConcurrencySafety:
    """Cancel used to blind-write a stale snapshot over a concurrent success."""

    @pytest.fixture
    def store(self, tmp_path: Path):
        store = JobStore(tmp_path / "store.sqlite3")
        yield store
        store.close()

    @staticmethod
    def _job(job_id: str = "job_a") -> JobRecord:
        return JobRecord(
            job_id=job_id,
            scene_id="scene_a",
            mode="reconstruct",
            state="running",
            created_at=utc_now(),
            updated_at=utc_now(),
            request={"mode": "reconstruct"},
            stages=[StageRecord(name="prepare"), StageRecord(name="export")],
        )

    def test_cancel_if_active_does_not_overwrite_a_finished_job(self, store: JobStore) -> None:
        job = self._job()
        store.insert_job(job)

        # The worker finishes first, publishing artifacts.
        from splat_api.app.jobstore import ArtifactRecord

        job.state = "succeeded"
        job.artifacts = [
            ArtifactRecord(
                name="splat.ply", kind="splat_ply", relative_path="output/splat.ply", size_bytes=1
            )
        ]
        for stage in job.stages:
            stage.state = "succeeded"
        store.update_job(job)

        # A cancel that read the row before that must not clobber it.
        result = store.cancel_if_active("job_a")
        assert result.state == "succeeded"
        assert [artifact.name for artifact in result.artifacts] == ["splat.ply"]
        assert store.get_job("job_a").state == "succeeded"

    def test_cancel_if_active_marks_a_live_job(self, store: JobStore) -> None:
        store.insert_job(self._job())
        result = store.cancel_if_active("job_a")
        assert result.state == "cancelled"
        assert all(stage.state == "cancelled" for stage in result.stages)

    def test_pagination_does_not_skip_same_millisecond_jobs(self, store: JobStore) -> None:
        """created_at is not unique; without a tiebreaker a page boundary lost rows."""
        stamp = utc_now()
        for index in range(5):
            job = self._job(f"job_{index:02d}")
            job.created_at = stamp
            job.updated_at = stamp
            store.insert_job(job)

        seen: list[str] = []
        cursor = None
        for _ in range(10):
            page = store.list_jobs(state=None, limit=2, before=cursor, scene_id=None)
            if not page:
                break
            seen.extend(job.job_id for job in page)
            cursor = f"{page[-1].created_at}|{page[-1].job_id}"
        assert sorted(seen) == [f"job_{index:02d}" for index in range(5)]

    def test_delete_job_removes_the_row(self, store: JobStore) -> None:
        store.insert_job(self._job())
        store.delete_job("job_a")
        assert store.count_active_jobs() == 0


@pytest.mark.usefixtures("fake_pipeline")
class TestProgressReporting:
    def test_a_failed_job_does_not_report_near_complete_progress(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skipped stages used to count as done, so a first-stage failure read 0.5."""

        def broken(settings, paths, **kwargs):
            return pipeline.StageCommand(
                name=pipeline.STAGE_PREPARE,
                argv=("/bin/sh", "-c", "exit 7"),
                description="failing stage",
            )

        monkeypatch.setattr(pipeline, "prepare_command", broken)
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
        finished = wait_for_job(client, job_id)
        assert finished["state"] == "failed"
        assert finished["progress"] == 0.0

    def test_error_message_carries_no_absolute_paths(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
    ) -> None:
        def broken(settings_arg, paths, **kwargs):
            return pipeline.StageCommand(
                name=pipeline.STAGE_PREPARE,
                argv=("/bin/sh", "-c", f"echo failed in {paths.prepared_root} >&2; exit 1"),
                description="failing stage",
            )

        monkeypatch.setattr(pipeline, "prepare_command", broken)
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
        finished = wait_for_job(client, job_id)
        assert finished["state"] == "failed"
        assert str(settings.data_root) not in finished["error"]
        assert "<data_root>" in finished["error"]


@pytest.mark.usefixtures("fake_pipeline")
class TestArtifactDownloadsAreNotRecompressed:
    def test_binary_artifacts_are_served_identity_encoded(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Gzipping a PLY on the event loop defeats sendfile and saves nothing."""
        archive, _ = build_colmap_zip(tmp_path)
        scene_id = upload_scene(client, archive)["scene_id"]
        job_id = client.post("/v1/jobs", json={"scene_id": scene_id}, headers=auth()).json()["job_id"]
        assert wait_for_job(client, job_id)["state"] == "succeeded"

        response = client.get(
            f"/v1/jobs/{job_id}/artifacts/splat.ply",
            headers={**auth(READ_KEY), "Accept-Encoding": "gzip, deflate"},
        )
        assert response.status_code == 200
        assert response.headers.get("content-encoding") is None
