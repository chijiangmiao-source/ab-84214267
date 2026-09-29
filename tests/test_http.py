"""End-to-end HTTP tests against the stdlib server (in-process sockets)."""

from __future__ import annotations

import base64
import json
import tempfile
import threading
import unittest
import urllib.request
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError

import _bootstrap  # noqa: F401

from elfaudit.fixture import R_X86_64_64, R_X86_64_PC32, build_object
from elfaudit.server import (
    ConclusionStore,
    build_server,
)


def _obj_dual() -> bytes:
    text = b"\x00" * 8 + b"\x00" * 4
    relocs = [
        (0x0, R_X86_64_64, "foo", 0),
        (0x8, R_X86_64_PC32, "bar", -4),
    ]
    return build_object(text=text, relocs=relocs, externals=["foo", "bar"])


def _obj_overlap() -> bytes:
    text = b"\x00" * 16
    relocs = [
        (0x0, R_X86_64_64, "foo", 0),
        (0x4, R_X86_64_PC32, "bar", 0),
    ]
    return build_object(text=text, relocs=relocs, externals=["foo", "bar"])


def _obj_pc32_overflow() -> bytes:
    text = b"\x00" * 4
    relocs = [(0x0, R_X86_64_PC32, "foo", 0)]
    return build_object(text=text, relocs=relocs, externals=["foo"])


class ServerHarness:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ConclusionStore(Path(self.tmp.name))
        self.httpd = build_server("127.0.0.1", 0, self.store, 16 << 20)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def request(self, path: str, payload: dict | None = None,
                method: str | None = None):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = method or "POST"
        req = urllib.request.Request(self.url(path), data=data,
                                     headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except HTTPError as e:
            return e.code, json.loads(e.read())


class HttpTests(unittest.TestCase):
    def test_healthz(self):
        with ServerHarness() as h:
            status, body = h.request("/healthz")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")

    def test_page_served(self):
        with ServerHarness() as h:
            with urllib.request.urlopen(h.url("/"), timeout=5) as resp:
                html = resp.read().decode("utf-8")
            self.assertEqual(resp.status, 200)
            self.assertIn("ELF64 重定位审计台", html)

    def test_externals_endpoint(self):
        with ServerHarness() as h:
            payload = {"file_base64": base64.b64encode(_obj_dual()).decode()}
            status, body = h.request("/api/externals", payload)
            self.assertEqual(status, 200)
            self.assertEqual(body["external_symbols"], ["foo", "bar"])
            self.assertEqual(body["reloc_count"], 2)

    def test_audit_success_freeze_and_fetch(self):
        with ServerHarness() as h:
            payload = {
                "audit_id": "REL-001",
                "load_base": "0x400000",
                "file_base64": base64.b64encode(_obj_dual()).decode(),
                "symbols": {"foo": "0x500000", "bar": "0x2000000"},
            }
            status, body = h.request("/api/audit", payload)
            self.assertEqual(status, 200, body)
            self.assertTrue(body["ok"])
            rec = body["record"]
            self.assertEqual(rec["conclusion"], "FROZEN")
            self.assertEqual(len(rec["patches"]), 2)
            self.assertEqual(rec["patches"][0]["S"], "0x500000")
            # PC32: 0x2000000 - 4 - (0x400000 + 8) = 0x1bfff f4
            self.assertEqual(rec["patches"][1]["value"],
                             f"0x{0x2000000 - 4 - (0x400000 + 8):x}")

            # Frozen record is retrievable later by id.
            status2, body2 = h.request("/api/audit?id=REL-001", method="GET")
            self.assertEqual(status2, 200)
            self.assertEqual(body2["record"]["summary"]["patch_count"], 2)

            # Stored on disk.
            self.assertTrue((Path(h.tmp.name) / "REL-001.json").exists())

    def test_overlap_rejected_clears_prior_success(self):
        with ServerHarness() as h:
            good = {
                "audit_id": "REL-002",
                "load_base": "0x400000",
                "file_base64": base64.b64encode(_obj_dual()).decode(),
                "symbols": {"foo": "0x500000", "bar": "0x2000000"},
            }
            status, body = h.request("/api/audit", good)
            self.assertEqual(status, 200, body)

            bad = dict(good)
            bad["file_base64"] = base64.b64encode(_obj_overlap()).decode()
            status, body = h.request("/api/audit", bad)
            self.assertEqual(status, 422, body)
            self.assertFalse(body["ok"])
            rec = body["record"]
            self.assertEqual(rec["conclusion"], "REJECTED")
            self.assertIn("重叠", rec["error"])
            self.assertIn("重定位项", rec["position"])

            # The old success conclusion must be gone for that id.
            _, fetched = h.request("/api/audit?id=REL-002", method="GET")
            self.assertEqual(fetched["record"]["conclusion"], "REJECTED")
            self.assertNotIn("patches", fetched["record"])

    def test_pc32_overflow_rejected_no_partial_result(self):
        with ServerHarness() as h:
            payload = {
                "audit_id": "REL-003",
                "load_base": "0x400000",
                "file_base64": base64.b64encode(_obj_pc32_overflow()).decode(),
                # ~ 0x400000 + INT32_MAX + 1 away
                "symbols": {"foo": hex(0x400000 + 0x7FFFFFFF + 1)},
            }
            status, body = h.request("/api/audit", payload)
            self.assertEqual(status, 422, body)
            rec = body["record"]
            self.assertEqual(rec["conclusion"], "REJECTED")
            self.assertIn("32 位范围", rec["error"])
            self.assertIn("重定位项[0]", rec["position"])

            _, fetched = h.request("/api/audit?id=REL-003", method="GET")
            self.assertEqual(fetched["record"]["conclusion"], "REJECTED")
            self.assertNotIn("patches", fetched["record"])

    def test_bad_base64_rejected(self):
        with ServerHarness() as h:
            status, body = h.request("/api/externals",
                                     {"file_base64": "@@not-base64@@"})
            self.assertEqual(status, 400)
            self.assertIn("Base64", body["error"])

    def test_bad_audit_id(self):
        with ServerHarness() as h:
            status, body = h.request("/api/audit",
                                     {"audit_id": "../escape",
                                      "file_base64": ""})
            self.assertEqual(status, 400)
            self.assertEqual(body["position"], "audit_id")

    def test_fetch_unknown_id(self):
        with ServerHarness() as h:
            status, body = h.request("/api/audit?id=NOPE", method="GET")
            self.assertEqual(status, 404)
            self.assertFalse(body["ok"])

    def test_non_elf_rejected_and_fetched_as_rejection(self):
        with ServerHarness() as h:
            payload = {
                "audit_id": "REL-004",
                "load_base": "0x1000",
                "file_base64": base64.b64encode(b"not an elf file at all").decode(),
                "symbols": {},
            }
            status, body = h.request("/api/audit", payload)
            self.assertEqual(status, 422, body)
            self.assertEqual(body["record"]["conclusion"], "REJECTED")
            self.assertTrue(body["record"]["position"])
            _, fetched = h.request("/api/audit?id=REL-004", method="GET")
            self.assertEqual(fetched["record"]["conclusion"], "REJECTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
