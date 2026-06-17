#!/usr/bin/env python3
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import requests


UPSTREAM = os.environ.get("OWUI_UPSTREAM", "http://127.0.0.1:8090").rstrip("/")
LISTEN_HOST = os.environ.get("WRAPPER_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("WRAPPER_PORT", "8080"))
ALLOWED_PATHS = frozenset(
    path.strip().rstrip("/")
    for path in os.environ.get(
        "WRAPPER_ALLOWED_PATHS",
        "/api/chat/completions,/api/v1/chat/completions,/openai/chat/completions,/openai/v1/chat/completions",
    ).split(",")
    if path.strip()
)
MAX_PATCH_BODY_BYTES = int(os.environ.get("WRAPPER_MAX_PATCH_BODY_BYTES", str(64 * 1024 * 1024)))

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

SESSION_HEADER_CANDIDATES = (
    "x-litellm-session-id",
    "x-session-affinity",
    "x-session-id",
    "x-opencode-session",
    "x-opencode-session-id",
    "x-codex-session-id",
    "x-codex-thread-id",
    "x-continue-session-id",
    "x-claude-code-session-id",
    "x-claude-session-id",
    "x-roo-root-task-id",
    "x-roo-task-id",
    "x-roo-code-root-task-id",
    "x-roo-code-task-id",
    "x-kilo-session",
    "x-kilocode-taskid",
    "x-kilocode-parent-taskid",
    "x-kilo-code-session-id",
    "session_id",
)


def clean(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "undefined"}:
        return None
    return text[:512]


def lower_headers(headers):
    return {str(key).lower(): value for key, value in headers.items()}


def find_session(headers, body):
    for name in SESSION_HEADER_CANDIDATES:
        value = clean(headers.get(name))
        if value:
            return value, name

    if isinstance(body, dict):
        metadata = body.get("metadata")
        for name, value in (
            ("metadata.chat_id", metadata.get("chat_id") if isinstance(metadata, dict) else None),
            ("chat_id", body.get("chat_id")),
            ("session_id", body.get("session_id")),
            ("thread_id", body.get("thread_id")),
            ("conversation_id", body.get("conversation_id")),
            ("id", body.get("id")),
        ):
            value = clean(value)
            if value:
                return value, name

    return None, None


def is_openai_chat_completion(path):
    normalized = path.rstrip("/")
    return normalized in {
        "/openai/chat/completions",
        "/openai/v1/chat/completions",
    }


def is_api_chat_completion(path):
    normalized = path.rstrip("/")
    return normalized in {
        "/api/chat/completions",
        "/api/v1/chat/completions",
    }


def normalize_upstream_path(path):
    if path.startswith("/openai/v1/"):
        return "/openai/" + path[len("/openai/v1/") :]
    return path


def patch_json_body(path, body_bytes, headers):
    content_type = (headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return body_bytes, False, None, None
    if len(body_bytes) > MAX_PATCH_BODY_BYTES:
        return body_bytes, False, None, "body_too_large"

    try:
        body = json.loads(body_bytes.decode("utf-8") if body_bytes else "{}")
    except Exception:
        return body_bytes, False, None, "invalid_json"

    if not isinstance(body, dict):
        return body_bytes, False, None, None

    session, source = find_session(headers, body)
    if not session:
        return body_bytes, False, None, None

    patched = False

    if is_openai_chat_completion(path):
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            body["metadata"] = metadata
        if not clean(metadata.get("chat_id")):
            metadata["chat_id"] = session
            patched = True

    elif is_api_chat_completion(path):
        if not clean(body.get("chat_id")):
            body["chat_id"] = session
            patched = True

    if not patched:
        return body_bytes, False, source, None

    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), True, source, None


class Wrapper(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.handle_method()

    def do_POST(self):
        self.handle_method()

    def do_PUT(self):
        self.handle_method()

    def do_PATCH(self):
        self.handle_method()

    def do_DELETE(self):
        self.handle_method()

    def do_OPTIONS(self):
        self.handle_method()

    def handle_method(self):
        if self.path == "/health":
            self.respond_json(200, {"status": True})
            return

        parsed = urlsplit(self.path)
        normalized_path = parsed.path.rstrip("/")
        if normalized_path not in ALLOWED_PATHS:
            self.respond_json(404, {"detail": "owui-sticky-wrapper proxies only chat completion endpoints"})
            return

        content_length = int(self.headers.get("content-length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b""
        request_headers = lower_headers(self.headers)

        patched = False
        source = None
        patch_error = None
        upstream_path = normalize_upstream_path(parsed.path)
        if self.command in {"POST", "PUT", "PATCH"} and (is_openai_chat_completion(parsed.path) or is_api_chat_completion(parsed.path)):
            body, patched, source, patch_error = patch_json_body(parsed.path, body, request_headers)

        path_with_query = upstream_path + (("?" + parsed.query) if parsed.query else "")
        upstream_url = UPSTREAM + path_with_query

        upstream_headers = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in {"host", "content-length", "x-smg-routing-key"}:
                continue
            upstream_headers[key] = value
        upstream_headers["Content-Length"] = str(len(body))
        upstream_headers["Accept-Encoding"] = "identity"
        upstream_headers["X-Forwarded-Host"] = self.headers.get("host", "")
        upstream_headers["X-Forwarded-Proto"] = self.headers.get("x-forwarded-proto", "http")
        upstream_headers["X-Forwarded-For"] = self.client_address[0]

        timeout = int(os.environ.get("WRAPPER_UPSTREAM_TIMEOUT", "1800"))
        response = None
        try:
            response = requests.request(
                self.command,
                upstream_url,
                data=body,
                headers=upstream_headers,
                stream=True,
                timeout=timeout,
            )
            self.send_response(response.status_code, response.reason)
            for key, value in response.headers.items():
                lower = key.lower()
                if lower in HOP_BY_HOP_HEADERS or lower in {"content-length"}:
                    continue
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.send_header("X-Sticky-Wrapper-Patched", "1" if patched else "0")
            if source:
                self.send_header("X-Sticky-Wrapper-Source", source)
            if patch_error:
                self.send_header("X-Sticky-Wrapper-Skip", patch_error)
            self.end_headers()

            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.respond_json(502, {"detail": "upstream proxy error", "error": str(exc)})
        finally:
            self.close_connection = True
            if response is not None:
                response.close()

    def respond_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def log_message(self, fmt, *args):
        sys.stdout.write("%s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()


def main():
    print(
        f"owui-sticky-wrapper listening on {LISTEN_HOST}:{LISTEN_PORT}, upstream={UPSTREAM}, paths={sorted(ALLOWED_PATHS)}",
        flush=True,
    )
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Wrapper).serve_forever()


if __name__ == "__main__":
    main()
