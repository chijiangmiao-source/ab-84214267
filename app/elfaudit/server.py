"""HTTP frontend: audit page, JSON API and health endpoint.

Stdlib-only so the container image needs no third-party packages.

Environment
-----------
ELFAUDIT_HOST      bind address (default 0.0.0.0)
ELFAUDIT_PORT      bind port    (default 8080)
ELFAUDIT_DATA      conclusion store directory (default ./data)
ELFAUDIT_MAX_BODY  max request body in bytes (default 16 MiB)
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .audit import audit_object, referenced_external_names
from .elf import UINT64_MASK, AuditError, parse_object

AUDIT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CONCLUSION_OK = "FROZEN"        # 冻结：通过
CONCLUSION_REJECTED = "REJECTED"
PREVIEW_LIMIT = 512

_STORE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# input helpers
# --------------------------------------------------------------------------- #
def _parse_uint64(value, field: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"{field} 必须是整数", position=field)
    if isinstance(value, int):
        n = value
        text = None
    elif isinstance(value, str):
        text = value.strip()
        try:
            n = int(text, 16) if text.lower().startswith(("0x", "-0x")) else int(text, 10)
        except ValueError:
            raise AuditError(f"{field} 不是合法整数：{value!r}", position=field)
    else:
        raise AuditError(f"{field} 必须是整数或字符串", position=field)
    if not 0 <= n <= UINT64_MASK:
        raise AuditError(f"{field}={text if text is not None else n} 超出 uint64 范围",
                         position=field)
    return n


def _decode_file(payload: dict) -> bytes:
    raw = payload.get("file_base64")
    if not isinstance(raw, str) or not raw.strip():
        raise AuditError("缺少 Base64 文件内容", position="file_base64")
    try:
        # validate=True rejects embedded/stray non-base64 characters.
        return base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AuditError(f"Base64 解码失败：{exc}", position="file_base64")


def _normalize_symbols(payload: dict) -> dict[str, int]:
    raw = payload.get("symbols", {})
    if not isinstance(raw, dict):
        raise AuditError("symbols 必须是 名称->地址 的对象", position="symbols")
    out: dict[str, int] = {}
    for name, addr in raw.items():
        if not isinstance(name, str) or not name:
            raise AuditError("外部符号名称必须为非空字符串", position="symbols")
        out[name] = _parse_uint64(addr, f"symbols.{name}")
    return out


# --------------------------------------------------------------------------- #
# conclusion store
# --------------------------------------------------------------------------- #
class ConclusionStore:
    """One frozen record per stable audit identifier."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, audit_id: str) -> Path:
        return self.root / f"{audit_id}.json"

    def save(self, audit_id: str, record: dict) -> None:
        tmp = self._path(audit_id).with_suffix(".json.tmp")
        with _STORE_LOCK:
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, self._path(audit_id))

    def load(self, audit_id: str) -> dict | None:
        with _STORE_LOCK:
            p = self._path(audit_id)
            if not p.exists():
                return None
            return json.loads(p.read_text(encoding="utf-8"))

    def clear_success(self, audit_id: str, rejection: dict) -> None:
        """Overwrite any prior frozen success with the new rejection."""
        self.save(audit_id, rejection)


# --------------------------------------------------------------------------- #
# application logic
# --------------------------------------------------------------------------- #
def make_record_ok(audit_id: str, load_base: int, data: bytes,
                   symbols: dict[str, int]) -> dict:
    result = audit_object(data, load_base, symbols)
    body = result.to_dict()
    patched_hex_whole = None
    return {
        "audit_id": audit_id,
        "conclusion": CONCLUSION_OK,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "file_size": body["file_size"],
            "file_sha256": body["file_sha256"],
            "text_size": body["text_size"],
            "load_base": body["load_base"],
            "original_text_sha256": body["original_text_sha256"],
            "patched_text_sha256": body["patched_text_sha256"],
            "external_symbols": body["external_symbols"],
            "patch_count": len(body["patches"]),
        },
        "patches": body["patches"],
    }


def make_record_rejected(audit_id: str, err: AuditError) -> dict:
    return {
        "audit_id": audit_id,
        "conclusion": CONCLUSION_REJECTED,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error": err.message,
        "position": err.position,
    }


def inspect_externals(data: bytes) -> dict:
    obj = parse_object(data)
    return {
        "external_symbols": referenced_external_names(obj),
        "text_size": len(obj.text_bytes),
        "reloc_count": len(obj.relocations),
    }


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
PAGE = (Path(__file__).with_name("static") / "index.html").read_text(
    encoding="utf-8"
)


class Handler(BaseHTTPRequestHandler):
    server_version = "elfaudit/1.0"

    # ---- helpers --------------------------------------------------------- #
    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        max_body = self.server.max_body  # type: ignore[attr-defined]
        if length <= 0:
            raise AuditError("缺少请求体", position="body")
        if length > max_body:
            raise AuditError(
                f"请求体 {length} 字节超过上限 {max_body}", position="Content-Length"
            )
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditError(f"JSON 解析失败：{exc}", position="body")
        if not isinstance(payload, dict):
            raise AuditError("请求体必须是 JSON 对象", position="body")
        return payload

    def log_message(self, fmt: str, *args) -> None:  # quiet, structured stderr
        print(f"[elfaudit] {self.address_string()} {fmt % args}", flush=True)

    # ---- routes ---------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok",
                                            "service": "elfaudit"})
            return
        if parsed.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/audit":
            qs = parse_qs(parsed.query)
            audit_id = (qs.get("id") or [""])[0]
            if not AUDIT_ID_RE.match(audit_id):
                self._send_json(HTTPStatus.BAD_REQUEST,
                                {"ok": False, "error": "非法审计标识",
                                 "position": "id"})
                return
            record = self.server.store.load(audit_id)  # type: ignore[attr-defined]
            if record is None:
                self._send_json(HTTPStatus.NOT_FOUND,
                                {"ok": False, "error": "该审计标识尚无冻结结论",
                                 "position": "id"})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "record": record})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "未知路径"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/externals":
                data = _decode_file(payload)
                self._send_json(HTTPStatus.OK, {"ok": True,
                                                **inspect_externals(data)})
                return
            if parsed.path == "/api/audit":
                audit_id = payload.get("audit_id")
                if not isinstance(audit_id, str) or not AUDIT_ID_RE.match(audit_id):
                    raise AuditError(
                        "审计标识必填，允许字母/数字/._-（1-64 字符，字母数字开头）",
                        position="audit_id",
                    )
                load_base = _parse_uint64(
                    payload.get("load_base", 0), "load_base"
                )
                data = _decode_file(payload)
                symbols = _normalize_symbols(payload)
                try:
                    record = make_record_ok(audit_id, load_base, data, symbols)
                    self.server.store.save(audit_id, record)  # type: ignore[attr-defined]
                    self._send_json(HTTPStatus.OK, {"ok": True, "record": record})
                except AuditError as err:
                    # First violation: freeze a rejection and make sure no
                    # stale success conclusion survives for this id.
                    rejection = make_record_rejected(audit_id, err)
                    self.server.store.clear_success(  # type: ignore[attr-defined]
                        audit_id, rejection
                    )
                    self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY,
                                    {"ok": False, "record": rejection})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "未知路径"})
        except AuditError as err:
            self._send_json(HTTPStatus.BAD_REQUEST,
                            {"ok": False, "error": err.message,
                             "position": err.position})
        except Exception as exc:  # defensive: never leak a traceback as 200
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                            {"ok": False, "error": f"服务器内部错误：{exc}"})


def build_server(host: str, port: int, store: ConclusionStore,
                 max_body: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.store = store           # type: ignore[attr-defined]
    httpd.max_body = max_body     # type: ignore[attr-defined]
    return httpd


def main() -> None:
    host = os.environ.get("ELFAUDIT_HOST", "0.0.0.0")
    port = int(os.environ.get("ELFAUDIT_PORT", "8080"))
    data_dir = Path(os.environ.get("ELFAUDIT_DATA", "data"))
    max_body = int(os.environ.get("ELFAUDIT_MAX_BODY", str(16 * 1024 * 1024)))
    store = ConclusionStore(data_dir)
    httpd = build_server(host, port, store, max_body)
    print(f"[elfaudit] listening on http://{host}:{port} (data={data_dir})",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
