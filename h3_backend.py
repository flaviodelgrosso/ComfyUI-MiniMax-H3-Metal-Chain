"""Persistent h3.c worker process and its JSONL protocol.

The worker (native/h3_comfy_worker) keeps one h3_ctx with h3.c's session cache
enabled, so the prepared model state survives between clips and between runs.
It lives in its own process: a native crash fails the node, not ComfyUI.
"""

import atexit
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque

NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
WORKER = os.path.join(NATIVE_DIR, "h3_comfy_worker")


class H3Error(RuntimeError):
    """Failure reported by h3.c or by the worker process."""


class H3Cancelled(H3Error):
    """The generation stopped because cancellation was requested."""


def check_environment():
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise H3Error("The h3.c backend needs macOS on Apple Silicon (Metal).")
    if not os.access(WORKER, os.X_OK):
        raise H3Error(f"The h3 worker is not built. Run: make -C '{NATIVE_DIR}' H3_DIR=/path/to/h3.c")
    for tool, override in (("ffmpeg", "H3_FFMPEG"), ("ffprobe", "H3_FFPROBE")):
        if not shutil.which(os.environ.get(override) or tool):
            raise H3Error(f"h3.c needs {tool} on PATH for MP4 output and image anchors "
                          f"(brew install ffmpeg, or set {override}).")


class H3Worker:
    def __init__(self):
        check_environment()
        self.process = subprocess.Popen([WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.stderr_tail = deque(maxlen=40)
        threading.Thread(target=self._drain_stderr, name="h3-worker-stderr", daemon=True).start()
        ready = self._exchange(None)
        logging.info(f"h3 worker started (pid {ready['pid']}, h3.c {ready['h3_version']} at {ready['h3_dir']})")

    def _drain_stderr(self):
        for raw in self.process.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            self.stderr_tail.append(line)
            logging.info(f"[h3] {line}")

    def alive(self):
        return self.process.poll() is None

    def _exchange(self, request, on_progress=None):
        """Send one request (None = only read the startup event) and return its final event."""
        cancel_sent = False
        try:
            if request is not None:
                self.process.stdin.write(json.dumps(request).encode() + b"\n")
                self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    code = self.process.wait()
                    tail = "\n".join(self.stderr_tail)
                    raise H3Error(f"h3 worker exited unexpectedly (code {code}).\n{tail}".rstrip())
                event = json.loads(line)
                kind = event["event"]
                if kind == "progress":
                    if on_progress is not None and on_progress(event["phase"], event["completed"], event["total"]) and not cancel_sent:
                        self.process.send_signal(signal.SIGUSR1)
                        cancel_sent = True
                elif kind == "error":
                    raise (H3Cancelled if event["cancelled"] else H3Error)(event["message"])
                else:
                    return event
        except H3Error:
            raise
        except BaseException:
            # The protocol state is unknown (broken pipe, bad line, exception in a callback): drop the worker.
            self.kill()
            raise

    def load(self, model_dir):
        """Load the model once; a repeated load of the same directory is a no-op in the worker."""
        return self._exchange({"command": "load", "model_dir": model_dir})

    def generate(self, request, on_progress=None):
        """Run one h3_generate(). on_progress(phase, completed, total) returning True cancels it."""
        return self._exchange({"command": "generate", **request}, on_progress)

    def kill(self):
        if self.alive():
            self.process.kill()
        self.process.wait()

    def close(self):
        if self.alive():
            try:
                self.process.stdin.close()  # EOF: the worker frees its h3_ctx and exits
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.kill()


_worker = None


def get_worker(model_dir):
    """The shared worker with model_dir loaded, (re)started if it is missing or dead."""
    global _worker
    if _worker is not None and not _worker.alive():
        logging.warning(f"h3 worker died (code {_worker.process.returncode}), restarting it")
        _worker = None
    if _worker is None:
        _worker = H3Worker()
    _worker.load(model_dir)
    return _worker


def shutdown_worker():
    global _worker
    if _worker is not None:
        _worker.close()
        _worker = None


atexit.register(shutdown_worker)
