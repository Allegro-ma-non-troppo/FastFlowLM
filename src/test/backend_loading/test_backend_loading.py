"""Exercise backend preflight through the installed CLI and REST entry points.

Use --preflight-only without model weights. Omit it and supply --active-model
plus --active-backend to also verify that refusals preserve a running model.
The script never requests a download of a compatible but uninstalled model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".github" / "script"))
from sweep_common import SweepTask, run_task


class BackendLoading(SweepTask):
    """Check backend override errors and active-model preservation."""

    name = "backend_loading"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--preflight-only", action="store_true")
        parser.add_argument("--active-model")
        parser.add_argument("--active-backend", choices=["flm", "core", "oga-hipep"])

    @staticmethod
    def offline_env():
        # A broken preflight must fail the test without downloading model weights.
        env = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "all_proxy"):
            env[key] = env[key.upper()] = "http://127.0.0.1:1"
        env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
        return env

    def post(self, path, payload):
        url = self.base_url.removesuffix("/v1") + path
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.args.request_timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return json.loads(response.read())

    def load_count(self):
        return self.server_log_path.read_text(encoding="utf-8", errors="replace").count("Loading model:")

    def generate(self, backend=None):
        payload = {
            "model": self.args.active_model,
            "messages": [{"role": "user", "content": "Reply with the number 2."}],
            "stream": False, "max_tokens": 8,
        }
        if backend:
            payload["backend"] = backend
        result = self.post("/v1/chat/completions", payload)
        if "error" in result or not result.get("choices"):
            raise AssertionError(result)

    def run(self):
        if self.args.no_server:
            raise ValueError("Use an isolated server so its load log can be checked")
        if not self.args.preflight_only:
            if not self.args.active_model or not self.args.active_backend:
                raise ValueError("Supply --active-model and --active-backend, or use --preflight-only")
            # Fail closed: the sweep's API catalog fallback cannot prove installation.
            listed = subprocess.run(
                [self.args.flm_bin, "list", "--json", "--filter", "installed"],
                check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                env=self.offline_env(),
            ).stdout
            entries = json.loads(listed[listed.index("{"):])["models"]
            installed = {entry.get("model") or entry.get("name") for entry in entries}
            if self.args.active_model not in installed:
                raise ValueError("The active model must already be installed")
            self.generate()
            loaded = self.load_count()
            self.generate(self.args.active_backend)
            if self.load_count() != loaded:
                raise AssertionError("Explicitly selecting the effective backend reloaded the model")

        # This shipped OGA tag has no native Q4NX entry point. Both frontends must
        # explain the missing file before they attempt a download or device load.
        incompatible = {"model": "qwen3.5-oga-hipep:9b", "backend": "flm", "stream": False}
        for path, content in [
            ("/api/generate", {"prompt": "Hello", "max_tokens": 1}),
            ("/api/chat", {"messages": [{"role": "user", "content": "Hello"}]}),
            ("/v1/completions", {"prompt": "Hello", "max_tokens": 1}),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1}),
        ]:
            result = self.post(path, incompatible | content)
            if "model.q4nx" not in json.dumps(result.get("error", "")):
                raise AssertionError(f"{path}: expected the package reason, got {result}")
            self.log_event("preflight_passed", detail=path)
        if self.load_count() != (loaded if not self.args.preflight_only else 0):
            raise AssertionError("A refused package caused a model load")

        if not self.args.preflight_only:
            self.generate()  # request override must not become the server default
            if self.load_count() != loaded:
                raise AssertionError("The original model was unloaded by a refusal")
            self.log_event("active_model_preserved", model=self.args.active_model)

        cli = subprocess.run(
            [self.args.flm_bin, "run", "qwen3.5-oga-hipep:9b", "--backend", "flm"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            env=self.offline_env(),
        )
        if cli.returncode == 0 or "model.q4nx" not in cli.stdout + cli.stderr:
            raise AssertionError("CLI did not refuse the incompatible package with its file reason")
        self.log_event("cli_preflight_passed")


if __name__ == "__main__":
    with patch.dict(os.environ, BackendLoading.offline_env()):
        raise SystemExit(run_task(BackendLoading))
