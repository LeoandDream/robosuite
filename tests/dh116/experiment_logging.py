"""Durable run logging for the nero_dh116 experiment.

Every run writes a JSONL file immediately on start and flushes each event to
disk. A run that is terminated before its final event remains visible with
``status=running``; normal exceptions and signals are recorded explicitly.
The project manifest provides a small before/after checksum audit for files
that belong to this experiment.
"""

import atexit
import hashlib
import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = Path(__file__).resolve().parent / "experiment_logs"
TRACKED_FILES = (
    "robosuite/models/assets/grippers/dh116.xml",
    "robosuite/models/assets/robots/nero/robot.xml",
    "robosuite/models/grippers/dh116_gripper.py",
    "robosuite/models/robots/manipulators/nero_robot.py",
    "robosuite/robots/robot.py",
    "robosuite/controllers/config/robots/default_nero.json",
    "tests/dh116/test_nero_dh116_lift.py",
    "tests/dh116/experiment_logging.py",
    "tests/dh116/nero_dh116.md",
    "tests/dh116/nero_dh116_technical.md",
    "tests/dh116/README.md",
    "tests/dh116/experiment_logs/project_changes.md",
)


def _json_default(value):
    """Serialize common numpy-like scalar values without importing numpy."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def project_manifest():
    """Return SHA-256 checksums for the files tracked by this experiment."""
    manifest = {}
    for relative_path in TRACKED_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            manifest[relative_path] = None
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[relative_path] = digest
    return manifest


class ExperimentLogger:
    """Append-only, interruption-visible logger for one experiment run."""

    def __init__(self, experiment, config=None):
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
        self.run_id = f"{stamp}_{os.getpid()}"
        self.path = LOG_ROOT / f"{experiment}_{self.run_id}.jsonl"
        self.latest_path = LOG_ROOT / "latest.json"
        self.experiment = experiment
        self.status = "running"
        self._closed = False
        self._old_handlers = {}
        self._before_manifest = project_manifest()

        self.event(
            "run_started",
            run_id=self.run_id,
            experiment=experiment,
            pid=os.getpid(),
            config=config or {},
            project_manifest_before=self._before_manifest,
        )
        self._write_latest(status="running")
        atexit.register(self._on_exit)
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._on_signal)

    def event(self, event, **data):
        """Write one event and synchronously flush it to the run log."""
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, status, result=None, error=None):
        """Close a run exactly once and write its after-run manifest."""
        if self._closed:
            return
        self.status = status
        self.event(
            "run_finished",
            run_id=self.run_id,
            status=status,
            result=result or {},
            error=error or {},
            project_manifest_after=project_manifest(),
        )
        self._write_latest(status=status, result=result, error=error)
        self._closed = True

    def _write_latest(self, status, result=None, error=None):
        payload = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "status": status,
            "log_file": str(self.path.relative_to(PROJECT_ROOT)),
            "result": result or {},
            "error": error or {},
        }
        self.latest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    def _on_signal(self, signum, _frame):
        self.finish(
            "interrupted",
            error={"signal": signum, "message": "run interrupted by signal"},
        )
        raise KeyboardInterrupt

    def _on_exit(self):
        if not self._closed:
            self.finish(
                "interrupted",
                error={"message": "process exited before run_finished was written"},
            )
