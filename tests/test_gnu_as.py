"""Cross-check against objects emitted by GNU as (skipped without binutils).

Builds a real ``.o`` with one R_X86_64_64 and one R_X86_64_PC32 against
undefined globals, then asserts the auditor's bytes match a manual
relocation of the section image.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from elfaudit.audit import audit_object

ASM = r"""
    .text
    .globl  ext64
    .globl  extpc
    .globl  func
    .type   func, @function
func:
    movabsq $ext64+7, %rax        # R_X86_64_64, addend 7 (8-byte slot)
    leaq    extpc-3(%rip), %rdx   # R_X86_64_PC32, addend -3 (4-byte slot)
    ret
    .size   func, .-func
    .section .note.GNU-stack,"",@progbits
"""


@unittest.skipUnless(shutil.which("as"), "GNU as 不可用")
class GnuAsCrossCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        src = self.dir / "t.s"
        obj = self.dir / "t.o"
        src.write_text(ASM)
        try:
            subprocess.run(["as", "--64", "-o", str(obj), str(src)],
                           check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            self.tmp.cleanup()
            self.skipTest(f"无可用的 x86-64 GNU as：{exc}")
        self.data = obj.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_object_relocations(self):
        base = 0x401000
        s64 = 0x7F000001
        spc = base + 0x100  # nearby, well within int32
        res = audit_object(self.data, base,
                           {"ext64": s64, "extpc": spc})
        kinds = sorted(p.reloc_name for p in res.patches)
        self.assertEqual(kinds, ["R_X86_64_64", "R_X86_64_PC32"])

        by_type = {p.reloc_name: p for p in res.patches}
        p64 = by_type["R_X86_64_64"]
        self.assertEqual(p64.a, 7)
        self.assertEqual(p64.value, s64 + 7)
        self.assertEqual(p64.after_hex, struct.pack("<Q", s64 + 7).hex())

        ppc = by_type["R_X86_64_PC32"]
        self.assertEqual(ppc.a, -3)
        expected = spc - 3 - (base + ppc.offset)
        self.assertEqual(ppc.value, expected)
        self.assertEqual(ppc.after_hex, struct.pack("<i", expected).hex())

        # The patched-image digest must equal an independent re-patch of
        # the raw .text bytes parsed from the file.
        from elfaudit.elf import parse_object

        text = bytearray(parse_object(self.data).text_bytes)
        for p in res.patches:
            width = p.width
            text[p.offset : p.offset + width] = bytes.fromhex(p.after_hex)
        import hashlib

        self.assertEqual(
            hashlib.sha256(bytes(text)).hexdigest(), res.patched_text_sha256
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
