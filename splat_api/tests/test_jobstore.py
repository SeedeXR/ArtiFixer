# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SQLite job/scene store, including restart behaviour."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from splat_api.app.errors import Conflict, NotFound
from splat_api.app.jobstore import (
    ArtifactRecord,
    JobRecord,
    JobStore,
    SceneRecord,
    StageRecord,
    utc_now,
)


@pytest.fixture
def store(tmp_path: Path):
    store = JobStore(tmp_path / "store.sqlite3")
    yield store
    store.close()


def make_scene(scene_id: str = "scene_a", digest: str | None = "d" * 64) -> SceneRecord:
    return SceneRecord(
        scene_id=scene_id,
        created_at=utc_now(),
        source="upload",
        summary={
            "image_count": 4,
            "camera_count": 1,
            "camera_models": ["PINHOLE"],
            "point_count": 10,
            "colmap_width": 64,
            "colmap_height": 48,
        },
        size_bytes=1234,
        digest=digest,
        image_names=["a.jpg", "b.jpg"],
    )


def make_job(job_id: str = "job_a", *, scene_id: str = "scene_a", token: str | None = None) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        scene_id=scene_id,
        mode="reconstruct",
        state="queued",
        created_at=utc_now(),
        updated_at=utc_now(),
        request={"mode": "reconstruct", "reconstruction_steps": 100},
        stages=[StageRecord(name="prepare"), StageRecord(name="export")],
        client_token=token,
    )


class TestScenes:
    def test_round_trips_a_scene(self, store: JobStore) -> None:
        scene = make_scene()
        store.insert_scene(scene)
        loaded = store.get_scene("scene_a")
        assert loaded.scene_id == scene.scene_id
        assert loaded.summary == scene.summary
        assert loaded.image_names == ["a.jpg", "b.jpg"]

    def test_rejects_duplicate_ids(self, store: JobStore) -> None:
        store.insert_scene(make_scene())
        with pytest.raises(Conflict, match="already exists"):
            store.insert_scene(make_scene())

    def test_unknown_scene_raises_not_found(self, store: JobStore) -> None:
        with pytest.raises(NotFound):
            store.get_scene("scene_missing")

    def test_lookup_by_digest_enables_upload_dedupe(self, store: JobStore) -> None:
        store.insert_scene(make_scene(digest="a" * 64))
        assert store.find_scene_by_digest("a" * 64) is not None
        assert store.find_scene_by_digest("b" * 64) is None

    def test_delete_is_blocked_while_jobs_are_active(self, store: JobStore) -> None:
        store.insert_scene(make_scene())
        store.insert_job(make_job())
        with pytest.raises(Conflict, match="active job"):
            store.delete_scene("scene_a")

    def test_delete_succeeds_once_jobs_are_terminal(self, store: JobStore) -> None:
        store.insert_scene(make_scene())
        job = make_job()
        store.insert_job(job)
        job.state = "succeeded"
        store.update_job(job)
        store.delete_scene("scene_a")
        with pytest.raises(NotFound):
            store.get_scene("scene_a")


class TestJobs:
    def test_round_trips_stages_and_artifacts(self, store: JobStore) -> None:
        job = make_job()
        job.artifacts = [
            ArtifactRecord(
                name="splat.ply",
                kind="splat_ply",
                relative_path="output/splat.ply",
                size_bytes=42,
                sha256="f" * 64,
            )
        ]
        job.stage("prepare").state = "succeeded"
        store.insert_job(job)

        loaded = store.get_job("job_a")
        assert isinstance(loaded.stages[0], StageRecord)
        assert loaded.stage("prepare").state == "succeeded"
        assert loaded.artifacts[0].name == "splat.ply"
        assert loaded.artifacts[0].sha256 == "f" * 64

    def test_progress_tracks_completed_stages(self, store: JobStore) -> None:
        job = make_job()
        assert job.progress == 0.0
        job.stage("prepare").state = "succeeded"
        assert job.progress == 0.5
        job.stage("export").state = "succeeded"
        assert job.progress == 1.0

    def test_client_token_is_unique(self, store: JobStore) -> None:
        store.insert_job(make_job("job_a", token="token-123"))
        with pytest.raises(Conflict, match="client_token"):
            store.insert_job(make_job("job_b", token="token-123"))

    def test_null_client_tokens_do_not_collide(self, store: JobStore) -> None:
        store.insert_job(make_job("job_a"))
        store.insert_job(make_job("job_b"))
        assert store.count_active_jobs() == 2

    def test_lookup_by_client_token(self, store: JobStore) -> None:
        store.insert_job(make_job("job_a", token="token-abc"))
        assert store.find_job_by_client_token("token-abc").job_id == "job_a"
        assert store.find_job_by_client_token("token-nope") is None

    def test_state_counts(self, store: JobStore) -> None:
        for index, state in enumerate(("queued", "running", "succeeded", "succeeded")):
            job = make_job(f"job_{index}")
            job.state = state
            store.insert_job(job)
        assert store.state_counts() == {"queued": 1, "running": 1, "succeeded": 2}


class TestPagination:
    def _seed(self, store: JobStore, count: int) -> list[JobRecord]:
        jobs = []
        for index in range(count):
            job = make_job(f"job_{index:02d}")
            store.insert_job(job)
            jobs.append(job)
            time.sleep(0.002)  # created_at has millisecond resolution
        return jobs

    def test_returns_newest_first(self, store: JobStore) -> None:
        self._seed(store, 5)
        listed = store.list_jobs(state=None, limit=10, before=None, scene_id=None)
        assert [job.job_id for job in listed] == [f"job_{index:02d}" for index in reversed(range(5))]

    def test_keyset_cursor_walks_the_whole_set(self, store: JobStore) -> None:
        self._seed(store, 7)
        seen: list[str] = []
        cursor = None
        while True:
            page = store.list_jobs(state=None, limit=3, before=cursor, scene_id=None)
            if not page:
                break
            seen.extend(job.job_id for job in page)
            cursor = page[-1].created_at
        assert sorted(seen) == [f"job_{index:02d}" for index in range(7)]
        assert len(seen) == len(set(seen))

    def test_filters_by_state_and_scene(self, store: JobStore) -> None:
        self._seed(store, 3)
        other = make_job("job_other", scene_id="scene_b")
        other.state = "failed"
        store.insert_job(other)

        assert [job.job_id for job in store.list_jobs(state="failed", limit=10, before=None, scene_id=None)] == [
            "job_other"
        ]
        assert [
            job.job_id for job in store.list_jobs(state=None, limit=10, before=None, scene_id="scene_b")
        ] == ["job_other"]


class TestRestartRecovery:
    def test_pending_and_running_jobs_are_returned_oldest_first(self, store: JobStore) -> None:
        queued = make_job("job_queued")
        time.sleep(0.002)
        running = make_job("job_running")
        running.state = "running"
        time.sleep(0.002)
        done = make_job("job_done")
        done.state = "succeeded"
        for job in (queued, running, done):
            store.insert_job(job)

        pending = store.pending_jobs_in_order()
        assert [job.job_id for job in pending] == ["job_queued", "job_running"]

    def test_records_survive_reopening_the_database(self, tmp_path: Path) -> None:
        path = tmp_path / "persist.sqlite3"
        first = JobStore(path)
        first.insert_scene(make_scene())
        first.insert_job(make_job())
        first.close()

        second = JobStore(path)
        try:
            assert second.get_job("job_a").scene_id == "scene_a"
            assert second.get_scene("scene_a").size_bytes == 1234
        finally:
            second.close()
