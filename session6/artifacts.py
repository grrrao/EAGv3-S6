from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from schemas import Artifact

_DEFAULT_DIR = Path("state/artifacts")


class ArtifactStore:
    def __init__(self, base_dir: Path = _DEFAULT_DIR):
        self._dir = base_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _stem(self, artifact_id: str) -> str:
        return artifact_id.removeprefix("art:")

    def put(self, blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
        sha = hashlib.sha256(blob).hexdigest()[:16]
        art_id = f"art:{sha}"
        stem = sha
        bin_path = self._dir / f"{stem}.bin"
        meta_path = self._dir / f"{stem}.json"
        if bin_path.exists():
            return art_id
        bin_path.write_bytes(blob)
        meta = Artifact(
            id=art_id,
            content_type=content_type,
            size_bytes=len(blob),
            source=source,
            descriptor=descriptor,
            created_at=datetime.now(timezone.utc),
        )
        meta_path.write_text(meta.model_dump_json(), encoding="utf-8")
        return art_id

    def get_bytes(self, artifact_id: str) -> bytes:
        return (self._dir / f"{self._stem(artifact_id)}.bin").read_bytes()

    def get_meta(self, artifact_id: str) -> Artifact:
        raw = (self._dir / f"{self._stem(artifact_id)}.json").read_text(encoding="utf-8")
        return Artifact.model_validate(json.loads(raw))

    def exists(self, artifact_id: str) -> bool:
        return (self._dir / f"{self._stem(artifact_id)}.bin").exists()
