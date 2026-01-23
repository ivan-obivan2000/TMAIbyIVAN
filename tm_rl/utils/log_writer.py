"""Log writer utilities based on TMInterface logging flow."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path


class WriterThread(threading.Thread):
    """Background writer that rotates log files on demand."""

    def __init__(self, q: queue.Queue, first_path: Path) -> None:
        super().__init__(daemon=True)
        self.q = q
        self.path = first_path
        self._stop = threading.Event()
        self._file = None

    def stop(self) -> None:
        self._stop.set()

    def _open(self, path: Path) -> None:
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
        self.path = path
        self._file = open(self.path, "w", encoding="utf-8")

    def run(self) -> None:
        self._open(self.path)

        while not self._stop.is_set():
            try:
                rec = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            if isinstance(rec, dict) and rec.get("_cmd") == "rotate":
                self._open(Path(rec["path"]))
                continue

            try:
                self._file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                # Drop bad writes without killing the thread.
                pass
