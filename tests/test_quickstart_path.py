"""Run the README quickstart for real, as a subprocess, against a stub upstream.

Every other test in this repository exercises the application through ASGI or
through imported functions. That leaves the actual thing a stranger does
untested: run the command in the README and point a client at it. A flag that no
longer exists, an entry point that fails to start, or a default that changed
would pass every other test and fail the first person who tries it.

This spawns the real CLI with the exact command the README gives, then sends the
request Claude Code sends, and asserts a well-formed Anthropic response comes
back and that the upstream saw an OpenAI Chat Completions call.

It does not cover llama.cpp or the model download, which are the parts of the
quickstart this repository does not own.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _make_upstream(received: list[dict]) -> type[BaseHTTPRequestHandler]:
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN002, D102
            return

        def do_POST(self):  # noqa: N802, D102
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            received.append({"path": self.path, "body": body})
            payload = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "kimi-linear"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "four bits is half of eight",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Upstream


def _wait_for(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True  # a 4xx still means the server is listening
        except OSError:
            time.sleep(0.4)
    return False


def test_readme_quickstart_serves_claude_code_through_to_an_openai_upstream() -> None:
    received: list[dict] = []
    upstream_port = _free_port()
    proxy_port = _free_port()

    server = HTTPServer(("127.0.0.1", upstream_port), _make_upstream(received))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # The command in README.md, with only the port changed so a developer's own
    # running instance cannot collide with the test.
    command = [
        sys.executable,
        "-m",
        "k3.cli",
        "serve",
        "--upstream",
        f"http://127.0.0.1:{upstream_port}/v1",
        "--model",
        "kimi-linear",
        "--reasoning-field",
        "inline",
        "--port",
        str(proxy_port),
    ]
    proxy = subprocess.Popen(
        command,
        cwd=str(REPOSITORY_ROOT),
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        if not _wait_for(f"http://127.0.0.1:{proxy_port}/v1/models", timeout=120):
            proxy.terminate()
            output = proxy.stdout.read() if proxy.stdout else ""
            pytest.fail(f"k3 serve never started listening:\n{output[:2000]}")

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "why does four bit quantisation save memory"}
                ],
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": "local",
                "anthropic-version": "2023-06-01",
                "user-agent": "claude-cli/1.0.0",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 200
            answer = json.loads(response.read())

        assert answer["type"] == "message"
        assert answer["role"] == "assistant"
        text = "".join(
            part.get("text", "") for part in answer["content"] if isinstance(part, dict)
        )
        assert "four bits" in text

        # The client spoke Anthropic Messages; the upstream must have been asked
        # in OpenAI Chat Completions. That translation is the whole product.
        assert len(received) == 1
        assert received[0]["path"].endswith("/chat/completions")
        assert received[0]["body"]["model"] == "kimi-linear"
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proxy.kill()
        server.shutdown()
        thread.join(timeout=10)
