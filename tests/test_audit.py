"""Audit core tests: parsing, relocation math, and every rejection path."""

from __future__ import annotations

import struct
import unittest

import _bootstrap  # noqa: F401  (sys.path setup)

from elfaudit.audit import audit_object
from elfaudit.elf import INT32_MAX, INT32_MIN, AuditError, parse_object
from elfaudit.fixture import R_X86_64_64, R_X86_64_PC32, build_object


BASE = 0x400000
S_FOO = 0x500000
S_BAR = 0x600000


def good_object() -> bytes:
    # .text: two 8-byte slots at 0x00/0x10 and one 4-byte slot at 0x18;
    # distinct, non-overlapping (section size 0x20).
    text = b"\x00" * 8 + b"\x90" * 8 + b"\x00" * 8 + b"\x00" * 4 + b"\xcc" * 4
    relocs = [
        (0x00, R_X86_64_64, "foo", 0x0),
        (0x10, R_X86_64_64, "foo", 0x1234),
        (0x18, R_X86_64_PC32, "bar", -0x10),
    ]
    return build_object(text=text, relocs=relocs, externals=["foo", "bar"])


class ParseRejectionTests(unittest.TestCase):
    def test_bad_magic(self):
        data = bytearray(good_object())
        data[0:4] = b"NOPE"
        with self.assertRaisesRegex(AuditError, "ELF 魔数") as cm:
            parse_object(bytes(data))
        self.assertEqual(cm.exception.position, "e_ident[0..3]")

    def test_bad_class(self):
        with self.assertRaisesRegex(AuditError, "ELF64"):
            parse_object(build_object(ei_class=1))

    def test_bad_endian(self):
        with self.assertRaisesRegex(AuditError, "小端"):
            parse_object(build_object(ei_data=2))

    def test_not_rel(self):
        with self.assertRaisesRegex(AuditError, "ET_REL"):
            parse_object(build_object(e_type=2))  # ET_EXEC

    def test_bad_machine(self):
        with self.assertRaisesRegex(AuditError, "X86_64"):
            parse_object(build_object(e_machine=3))  # EM_386

    def test_truncated_header(self):
        with self.assertRaisesRegex(AuditError, "ELF 头|标识"):
            parse_object(b"\x7fELF\x02\x01\x01" + b"\x00" * 10)

    def test_section_table_truncated(self):
        data = bytearray(good_object())
        # Claim a section-table start 20 bytes past EOF.
        struct.pack_into("<Q", data, 40, len(data) - 20)
        with self.assertRaisesRegex(AuditError, "节头表"):
            parse_object(bytes(data))

    def test_shstrtab_wrong_type(self):
        # Point e_shstrndx at .symtab (type SYMTAB).
        data = bytearray(good_object())
        struct.pack_into("<H", data, 62, 3)
        with self.assertRaisesRegex(AuditError, "STRTAB"):
            parse_object(bytes(data))

    def test_reserved_section0_fields(self):
        data = bytearray(good_object())
        shoff = struct.unpack_from("<Q", data, 40)[0]
        struct.pack_into("<I", data, shoff + 4, 1)  # sh_type != NULL
        with self.assertRaisesRegex(AuditError, "保留节 0"):
            parse_object(bytes(data))

    def test_duplicate_text(self):
        with self.assertRaisesRegex(AuditError, r"一个名为 \.text"):
            parse_object(build_object(extra_text=True))

    def test_duplicate_rela(self):
        with self.assertRaisesRegex(AuditError, "一个 RELA"):
            parse_object(build_object(extra_rela=True))

    def test_rela_not_pointing_at_text(self):
        with self.assertRaisesRegex(AuditError, "sh_info"):
            parse_object(build_object(rela_sh_info=5))

    def test_rela_link_not_symtab(self):
        with self.assertRaisesRegex(AuditError, "SYMTAB"):
            parse_object(build_object(rela_sh_link=4))

    def test_symtab_link_not_strtab(self):
        with self.assertRaisesRegex(AuditError, "STRTAB"):
            parse_object(build_object(symtab_sh_link=1))

    def test_sht_rel_rejected(self):
        with self.assertRaisesRegex(AuditError, "RELA"):
            audit_object(build_object(with_rel=True), BASE, {})

    def test_reloc_table_truncated(self):
        data = bytearray(good_object())
        shoff = struct.unpack_from("<Q", data, 40)[0]
        # Shrink .rela.text sh_size (section index 2) by one byte.
        struct.pack_into("<Q", data, shoff + 2 * 64 + 32, 24 * 3 - 1)
        with self.assertRaisesRegex(AuditError, "RELA 节大小"):
            parse_object(bytes(data))

    def test_symtab_truncated(self):
        data = bytearray(good_object())
        shoff = struct.unpack_from("<Q", data, 40)[0]
        # Section 3 = .symtab; corrupt sh_size.
        struct.pack_into("<Q", data, shoff + 3 * 64 + 32, 24 * 3 - 1)
        with self.assertRaisesRegex(AuditError, "符号表大小"):
            parse_object(bytes(data))

    def test_nonzero_reserved_symbol(self):
        data = bytearray(good_object())
        # .symtab is section 3; first entry follows at its sh_offset.
        shoff = struct.unpack_from("<Q", data, 40)[0]
        sym_off = struct.unpack_from("<Q", data, shoff + 3 * 64 + 24)[0]
        data[sym_off + 4] = 0x10  # st_info != 0
        with self.assertRaisesRegex(AuditError, "符号表第 0 项"):
            parse_object(bytes(data))

    def test_text_shorter_than_file_range(self):
        data = bytearray(good_object())
        shoff = struct.unpack_from("<Q", data, 40)[0]
        struct.pack_into("<Q", data, shoff + 1 * 64 + 32, 0xFFFFFFFF)
        with self.assertRaisesRegex(AuditError, r"\.text.*数据|越界"):
            parse_object(bytes(data))


class AuditSuccessTests(unittest.TestCase):
    def test_dual_type_relocations(self):
        res = audit_object(good_object(), BASE, {"foo": S_FOO, "bar": S_BAR})
        self.assertEqual(len(res.patches), 3)
        offsets = [p.offset for p in res.patches]
        self.assertEqual(offsets, sorted(offsets))

        p0 = res.patches[0]
        self.assertEqual(p0.reloc_name, "R_X86_64_64")
        self.assertEqual(p0.s, S_FOO)
        self.assertEqual(p0.a, 0)
        self.assertEqual(p0.p, BASE)
        self.assertEqual(p0.value, S_FOO)
        self.assertEqual(p0.before_hex, "00" * 8)
        self.assertEqual(p0.after_hex, S_FOO.to_bytes(8, "little").hex())

        p1 = res.patches[1]
        self.assertEqual(p1.value, S_FOO + 0x1234)
        self.assertEqual(p1.before_hex, "00" * 8)
        self.assertEqual(
            p1.after_hex, (S_FOO + 0x1234).to_bytes(8, "little").hex()
        )

        p2 = res.patches[2]
        self.assertEqual(p2.reloc_name, "R_X86_64_PC32")
        expected = S_BAR - 0x10 - (BASE + 0x18)
        self.assertEqual(p2.value, expected)
        self.assertEqual(p2.width, 4)
        self.assertEqual(p2.after_hex, struct.pack("<i", expected).hex())
        self.assertEqual(p2.before_hex, "00000000")

        self.assertEqual(len(res.original_text_sha256), 64)
        self.assertNotEqual(res.original_text_sha256, res.patched_text_sha256)
        self.assertEqual(res.external_symbols, ("foo", "bar"))

    def test_pc32_negative_value_packed_twos_complement(self):
        text = b"\x00" * 4
        # S far below P, with a negative result that must be sign-encoded.
        relocs = [(0x0, R_X86_64_PC32, "foo", 0)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        res = audit_object(data, 0x1000, {"foo": 0x0})
        value = res.patches[0].value
        self.assertEqual(value, -0x1000)
        self.assertEqual(res.patches[0].after_hex,
                         struct.pack("<i", -0x1000).hex())

    def test_pc32_boundary_values_accepted(self):
        text = b"\x00" * 8
        # INT32_MIN at offset 0 and INT32_MAX at offset 4.
        relocs = [
            (0x0, R_X86_64_PC32, "foo", 0),
            (0x4, R_X86_64_PC32, "bar", 0),
        ]
        data = build_object(text=text, relocs=relocs, externals=["foo", "bar"])
        s_foo = BASE + INT32_MIN & 0xFFFFFFFFFFFFFFFF
        s_bar = (BASE + 4 + INT32_MAX) & 0xFFFFFFFFFFFFFFFF
        res = audit_object(data, BASE, {"foo": s_foo, "bar": s_bar})
        self.assertEqual(res.patches[0].value, INT32_MIN)
        self.assertEqual(res.patches[1].value, INT32_MAX)

    def test_r64_with_negative_addend(self):
        text = b"\xff" * 8
        relocs = [(0x0, R_X86_64_64, "foo", -0x10)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        res = audit_object(data, BASE, {"foo": 0x20})
        self.assertEqual(res.patches[0].value, 0x10)
        self.assertEqual(res.patches[0].before_hex, "ff" * 8)

    def test_sorted_by_offset_even_if_table_is_not(self):
        text = b"\x00" * 24
        relocs = [
            (0x10, R_X86_64_64, "foo", 0),
            (0x00, R_X86_64_64, "bar", 0),
        ]
        data = build_object(text=text, relocs=relocs, externals=["foo", "bar"])
        res = audit_object(data, BASE, {"foo": S_FOO, "bar": S_BAR})
        self.assertEqual([p.offset for p in res.patches], [0x00, 0x10])


class AuditRejectionTests(unittest.TestCase):
    def _audit(self, data, symbols=None, base=BASE):
        return audit_object(data, base, symbols if symbols is not None
                            else {"foo": S_FOO, "bar": S_BAR})

    def test_unsupported_reloc_type(self):
        text = b"\x00" * 8
        # type 10 (R_X86_64_32PL) is not in the accepted set.
        relocs = [(0x0, 10, "foo", 0)]
        with self.assertRaisesRegex(AuditError, "不支持的重定位类型") as cm:
            self._audit(build_object(text=text, relocs=relocs,
                                     externals=["foo"]), {"foo": S_FOO})
        self.assertIn("r_info.type", cm.exception.position)

    def test_symbol_index_out_of_range(self):
        text = b"\x00" * 8
        relocs = [(0x0, R_X86_64_64, 99, 0)]
        with self.assertRaisesRegex(AuditError, "符号索引 99 越界") as cm:
            self._audit(build_object(text=text, relocs=relocs))
        self.assertIn("重定位项[0]", cm.exception.position)

    def test_symbol_index_zero(self):
        text = b"\x00" * 8
        relocs = [(0x0, R_X86_64_64, 0, 0)]
        with self.assertRaisesRegex(AuditError, "符号索引 0"):
            self._audit(build_object(text=text, relocs=relocs))

    def test_defined_symbol_rejected(self):
        text = b"\x00" * 8
        relocs = [(0x0, R_X86_64_64, "loc", 0)]
        data = build_object(text=text, relocs=relocs,
                            externals=[], defined_symbol="loc")
        with self.assertRaisesRegex(AuditError, "非外部符号"):
            self._audit(data, {})

    def test_missing_symbol_address(self):
        text = b"\x00" * 8
        relocs = [(0x0, R_X86_64_64, "foo", 0)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        with self.assertRaisesRegex(AuditError, "缺少外部符号 'foo' 的地址"):
            audit_object(data, BASE, {})

    def test_unexpected_symbol_address(self):
        data = good_object()
        with self.assertRaisesRegex(AuditError, "未声明的外部符号"):
            audit_object(data, BASE,
                         {"foo": S_FOO, "bar": S_BAR, "ghost": 0x1})

    def test_write_range_overflow_r64(self):
        text = b"\x00" * 8
        relocs = [(0x4, R_X86_64_64, "foo", 0)]  # 0x4+8 > 8
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        with self.assertRaisesRegex(AuditError, "写入区间") as cm:
            audit_object(data, BASE, {"foo": S_FOO})
        self.assertIn("r_offset", cm.exception.position)

    def test_write_range_at_end_r64(self):
        # r_offset == size must be rejected (no bytes to write).
        text = b"\x00" * 8
        relocs = [(0x8, R_X86_64_64, "foo", 0)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        with self.assertRaisesRegex(AuditError, "写入区间"):
            audit_object(data, BASE, {"foo": S_FOO})

    def test_write_range_pc32(self):
        text = b"\x00" * 4
        relocs = [(0x2, R_X86_64_PC32, "foo", 0)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        with self.assertRaisesRegex(AuditError, "写入区间"):
            audit_object(data, BASE, {"foo": S_FOO})

    def test_overlapping_writes_r64_and_pc32(self):
        # 8-byte slot at 0x00 and 4-byte slot at 0x04 overlap.
        text = b"\x00" * 16
        relocs = [
            (0x00, R_X86_64_64, "foo", 0),
            (0x04, R_X86_64_PC32, "bar", 0),
        ]
        data = build_object(text=text, relocs=relocs, externals=["foo", "bar"])
        with self.assertRaisesRegex(AuditError, "补丁写入区间重叠") as cm:
            audit_object(data, BASE, {"foo": S_FOO, "bar": S_BAR})
        self.assertIn("重定位项[1]", cm.exception.position)

    def test_adjacent_writes_do_not_overlap(self):
        text = b"\x00" * 12
        relocs = [
            (0x00, R_X86_64_64, "foo", 0),
            (0x08, R_X86_64_PC32, "bar", 0),
        ]
        data = build_object(text=text, relocs=relocs, externals=["foo", "bar"])
        res = audit_object(data, 0x1000000, {"foo": 0x2000000, "bar": 0x1000008})
        self.assertEqual(len(res.patches), 2)

    def test_pc32_overflow_positive_no_partial_result(self):
        text = b"\x00" * 16
        relocs = [
            (0x00, R_X86_64_PC32, "foo", 0),
            (0x08, R_X86_64_64, "bar", 0),
        ]
        data = build_object(text=text, relocs=relocs, externals=["foo", "bar"])
        # S - P = INT32_MAX + 1.
        s_foo = (BASE + INT32_MAX + 1) & 0xFFFFFFFFFFFFFFFF
        try:
            audit_object(data, BASE, {"foo": s_foo, "bar": S_BAR})
        except AuditError as err:
            self.assertIn("有符号 32 位范围", err.message)
            self.assertIn("重定位项[0]", err.position)
        else:
            self.fail("PC32 溢出必须被拒绝")

    def test_pc32_overflow_negative(self):
        text = b"\x00" * 4
        relocs = [(0x00, R_X86_64_PC32, "foo", 0)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        s_foo = (BASE + INT32_MIN - 1) & 0xFFFFFFFFFFFFFFFF
        with self.assertRaisesRegex(AuditError, "有符号 32 位范围"):
            audit_object(data, BASE, {"foo": s_foo})

    def test_r64_underflow_rejected(self):
        text = b"\x00" * 8
        relocs = [(0x00, R_X86_64_64, "foo", -1)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        with self.assertRaisesRegex(AuditError, "uint64"):
            audit_object(data, BASE, {"foo": 0})

    def test_r64_overflow_rejected(self):
        text = b"\x00" * 8
        relocs = [(0x00, R_X86_64_64, "foo", 1)]
        data = build_object(text=text, relocs=relocs, externals=["foo"])
        with self.assertRaisesRegex(AuditError, "uint64"):
            audit_object(data, BASE, {"foo": 0xFFFFFFFFFFFFFFFF})

    def test_invalid_load_base(self):
        with self.assertRaisesRegex(AuditError, "装载基址"):
            audit_object(good_object(), -1, {"foo": S_FOO, "bar": S_BAR})
        with self.assertRaisesRegex(AuditError, "装载基址"):
            audit_object(good_object(), 1 << 64, {"foo": S_FOO, "bar": S_BAR})

    def test_invalid_symbol_address(self):
        with self.assertRaisesRegex(AuditError, "uint64"):
            audit_object(good_object(), BASE,
                         {"foo": "nope", "bar": S_BAR})  # type: ignore[dict-item]

    def test_first_violation_position_is_reported(self):
        # Table order differs from offset order; the overlap diagnostic
        # must identify the entry at the later offset (table index 0).
        text = b"\x00" * 16
        relocs = [
            (0x04, R_X86_64_PC32, "bar", 0),
            (0x00, R_X86_64_64, "foo", 0),
        ]
        data = build_object(text=text, relocs=relocs, externals=["foo", "bar"])
        with self.assertRaises(AuditError) as cm:
            audit_object(data, BASE, {"foo": S_FOO, "bar": S_BAR})
        self.assertIn("重定位项[0]", cm.exception.position)


if __name__ == "__main__":
    unittest.main(verbosity=2)
