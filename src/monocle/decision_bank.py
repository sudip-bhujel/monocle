from __future__ import annotations

import csv
from datetime import UTC, datetime
import fcntl
import json
import os
from pathlib import Path
import tempfile

from monocle.schema import Decision


FIELDS = ["request_fingerprint", "request_json", "decision_json", "created_at"]


class DecisionBank:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._lock = None
        self._records: dict[str, dict[str, str]] = {}
        self._dirty = False
        self._puts_since_flush = 0

    def __enter__(self) -> DecisionBank:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            self._lock = None
            raise RuntimeError(
                f"decision bank is already in use by another process: {self.path}"
            ) from error
        self._records = self._read()
        self._puts_since_flush = 0
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            # Flush even on KeyboardInterrupt so partial runs remain resumable.
            if self._dirty:
                self._write()
        finally:
            if self._lock is not None:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
                self._lock.close()
                self._lock = None

    def get(self, fingerprint: str) -> Decision | None:
        record = self._records.get(fingerprint)
        if record is None:
            return None
        decision = Decision.model_validate_json(record["decision_json"])
        if decision.raw_label == "error" or decision.error:
            raise ValueError(f"decision bank contains an error response: {fingerprint}")
        return decision

    def put(self, fingerprint: str, request_json: str, decision: Decision) -> None:
        if decision.raw_label == "error" or decision.error:
            return
        if fingerprint in self._records:
            return
        self._records[fingerprint] = {
            "request_fingerprint": fingerprint,
            "request_json": request_json,
            "decision_json": decision.model_dump_json(exclude_none=True),
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._dirty = True
        self._puts_since_flush += 1
        if self._puts_since_flush >= 25:
            self._write()
            self._dirty = False
            self._puts_since_flush = 0

    def _read(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError(
                    f"invalid decision-bank columns in {self.path}: {reader.fieldnames}"
                )
            records = {}
            for row in reader:
                fingerprint = row["request_fingerprint"]
                if not fingerprint or fingerprint in records:
                    raise ValueError(
                        f"duplicate or empty request fingerprint in {self.path}"
                    )
                json.loads(row["request_json"])
                decision = Decision.model_validate_json(row["decision_json"])
                if decision.raw_label == "error" or decision.error:
                    raise ValueError(
                        f"decision bank contains an error response: {fingerprint}"
                    )
                records[fingerprint] = row
        return records

    def _write(self) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(
                    self._records[key] for key in sorted(self._records)
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
