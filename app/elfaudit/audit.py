"""Relocation computation and end-to-end audit rules.

The audit is all-or-nothing: any single violation aborts and *no* patch
set is produced.  Each :class:`Patch` exposes S, A, P, the computed
value and the bytes before/after the write, ordered by section offset,
so a reviewer can independently recompute every result.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from .elf import (
    INT32_MAX,
    INT32_MIN,
    R_X86_64_64,
    R_X86_64_PC32,
    SUPPORTED_RELOC_TYPES,
    UINT64_MASK,
    AuditError,
    ElfObject,
    parse_object,
)

SHT_REL = 9  # rejected: only RELA carry explicit addends

RELOC_NAMES = {R_X86_64_64: "R_X86_64_64", R_X86_64_PC32: "R_X86_64_PC32"}
RELOC_WIDTH = {R_X86_64_64: 8, R_X86_64_PC32: 4}


@dataclass(frozen=True)
class Patch:
    reloc_index: int
    reloc_type: int
    reloc_name: str
    symbol: str
    offset: int  # r_offset inside .text
    width: int
    s: int  # external symbol address
    a: int  # r_addend
    p: int  # load_base + r_offset (PC of the patch slot)
    value: int  # computed value written into the slot
    before_hex: str
    after_hex: str


@dataclass(frozen=True)
class AuditResult:
    text_size: int
    original_text_sha256: str
    patched_text_sha256: str
    file_sha256: str
    file_size: int
    load_base: int
    external_symbols: tuple[str, ...]
    patches: tuple[Patch, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "text_size": self.text_size,
            "original_text_sha256": self.original_text_sha256,
            "patched_text_sha256": self.patched_text_sha256,
            "file_sha256": self.file_sha256,
            "file_size": self.file_size,
            "load_base": _hex(self.load_base),
            "external_symbols": list(self.external_symbols),
            "patches": [
                {
                    "reloc_index": p.reloc_index,
                    "type": p.reloc_name,
                    "symbol": p.symbol,
                    "offset": _hex(p.offset),
                    "width": p.width,
                    "S": _hex(p.s),
                    "A": _hex_signed(p.a),
                    "P": _hex(p.p),
                    "value": _hex_signed(p.value),
                    "before": p.before_hex,
                    "after": p.after_hex,
                }
                for p in self.patches
            ],
        }


def _hex(v: int) -> str:
    return f"0x{v:x}"


def _hex_signed(v: int) -> str:
    return f"-0x{-v:x}" if v < 0 else f"0x{v:x}"


def referenced_external_names(obj: ElfObject) -> list[str]:
    """Names of undefined symbols actually referenced by relocations,
    in first-reference order."""
    names: list[str] = []
    seen: set[str] = set()
    for reloc in obj.relocations:
        sym_idx = reloc.sym_index
        if sym_idx == 0 or sym_idx >= len(obj.symbols):
            raise AuditError(
                f"r_info 符号索引 {sym_idx} 越界（符号表共 {len(obj.symbols)} 项）",
                position=f"重定位项[{reloc.index}].r_info.sym",
            )
        sym = obj.symbols[sym_idx]
        if not sym.is_undefined:
            continue  # reported per-relocation below
        if not sym.name:
            raise AuditError(
                f"外部符号[{sym_idx}] 名称为空", position=f"符号[{sym_idx}].st_name"
            )
        if sym.name not in seen:
            seen.add(sym.name)
            names.append(sym.name)
    # Two distinct undefined symbol entries sharing one name would make
    # a single supplied address ambiguous.
    name_to_indices: dict[str, list[int]] = {}
    for sym in obj.symbols:
        if sym.index != 0 and sym.is_undefined and sym.name in seen:
            name_to_indices.setdefault(sym.name, []).append(sym.index)
    for name, indices in name_to_indices.items():
        if len(indices) > 1:
            raise AuditError(
                f"外部符号名称 {name!r} 对应多个符号表项 {indices}，"
                "无法唯一确定其地址",
                position=f"符号[{indices[1]}].st_name",
            )
    return names


def _check_unsupported_sections(obj: ElfObject) -> None:
    for sec in obj.sections:
        if sec.sh_type == SHT_REL:
            raise AuditError(
                f"存在 SHT_REL 节[{sec.index}]({sec.name})，审计仅接受 RELA 重定位节",
                position=f"节[{sec.index}].sh_type",
            )


def audit_object(
    data: bytes,
    load_base: int,
    external_addrs: dict[str, int],
) -> AuditResult:
    """Validate every table/relocation and compute the frozen patch set.

    Raises :class:`AuditError` (with ``position``) on the first violation.
    On success returns patches sorted by offset; on failure nothing is
    returned (callers must not persist partial results).
    """
    if not isinstance(load_base, int) or isinstance(load_base, bool):
        raise AuditError("装载基址必须为整数", position="load_base")
    if not 0 <= load_base <= UINT64_MASK:
        raise AuditError("装载基址超出 uint64 范围", position="load_base")
    if not isinstance(external_addrs, dict):
        raise AuditError("外部符号地址必须为名称->地址的映射", position="symbols")

    obj = parse_object(data)
    _check_unsupported_sections(obj)
    external_names = referenced_external_names(obj)

    # P for the last byte of .text must remain inside the 64-bit address
    # space; otherwise every PC-relative slot is meaningless.
    text_size_hint = len(obj.text_bytes)
    if load_base + text_size_hint > UINT64_MASK:
        raise AuditError(
            f"装载基址 {_hex(load_base)} + .text 大小 {_hex(text_size_hint)} "
            "超出 uint64 地址空间",
            position="load_base",
        )

    # The supplied address map must only name symbols the object actually
    # declares external; typos must not silently audit the wrong target.
    known = set(external_names)
    for name in external_addrs:
        if not isinstance(name, str) or name not in known:
            raise AuditError(
                f"提供了对象未声明的外部符号地址：{name!r}", position=f"symbols.{name}"
            )
    for name, addr in external_addrs.items():
        if not isinstance(addr, int) or isinstance(addr, bool) or not 0 <= addr <= UINT64_MASK:
            raise AuditError(
                f"符号 {name!r} 的地址不是合法 uint64", position=f"symbols.{name}"
            )

    text = bytearray(obj.text_bytes)
    text_size = len(text)
    staged: list[tuple[int, int, bytes, Patch]] = []

    for reloc in obj.relocations:
        pos = f"重定位项[{reloc.index}]"
        if reloc.reloc_type not in SUPPORTED_RELOC_TYPES:
            raise AuditError(
                f"不支持的重定位类型 {reloc.reloc_type}（仅接受 "
                f"R_X86_64_64={R_X86_64_64} 与 R_X86_64_PC32={R_X86_64_PC32}）",
                position=f"{pos}.r_info.type",
            )

        # --- symbol table / index validation -----------------------------
        sym_idx = reloc.sym_index
        if sym_idx == 0 or sym_idx >= len(obj.symbols):
            raise AuditError(
                f"r_info 符号索引 {sym_idx} 越界（符号表共 {len(obj.symbols)} 项）",
                position=f"{pos}.r_info.sym",
            )
        sym = obj.symbols[sym_idx]
        if not sym.is_undefined:
            # The audit input only carries addresses for *external*
            # symbols; a defined target cannot be resolved here.
            raise AuditError(
                f"重定位引用了非外部符号[{sym_idx}]（{sym.name or '<无名>'}，"
                f"shndx={sym.shndx}），仅可审计外部符号",
                position=f"{pos}.r_info.sym",
            )
        if sym.name not in external_addrs:
            raise AuditError(
                f"缺少外部符号 {sym.name!r} 的地址", position=f"symbols.{sym.name}"
            )

        width = RELOC_WIDTH[reloc.reloc_type]
        r_off = reloc.offset
        end = r_off + width

        # --- write-range validation --------------------------------------
        if r_off > text_size or end > text_size:
            raise AuditError(
                f"写入区间 [{_hex(r_off)}, {_hex(end)}) 超出 .text 节大小 "
                f"{_hex(text_size)}",
                position=f"{pos}.r_offset",
            )

        s_addr = external_addrs[sym.name]
        addend = reloc.addend  # already structurally validated int64
        p_addr = (load_base + r_off) & UINT64_MASK

        if reloc.reloc_type == R_X86_64_64:
            # S + A, must be representable as unsigned 64 bits (no wrap).
            value = s_addr + addend
            if not 0 <= value <= UINT64_MASK:
                raise AuditError(
                    f"S + A = {_hex_signed(value)} 超出 uint64 无符号范围，"
                    "拒绝截断写入",
                    position=f"{pos}.computed",
                )
            after = struct.pack("<Q", value)
        else:
            # S + A - P, computed with 64-bit unsigned wraparound per the
            # x86-64 psABI, then required to fit a signed 32-bit slot.
            value_u = (s_addr + addend - p_addr) & UINT64_MASK
            if value_u >= 1 << 63:
                value = value_u - (1 << 64)
            else:
                value = value_u
            if not INT32_MIN <= value <= INT32_MAX:
                raise AuditError(
                    f"S + A - P = {_hex_signed(value)}（64 位无符号中间值 "
                    f"{_hex(value_u)}）超出有符号 32 位范围 "
                    f"[{_hex_signed(INT32_MIN)}, {_hex(INT32_MAX)}]，"
                    "拒绝生成任何补丁",
                    position=f"{pos}.computed",
                )
            after = struct.pack("<i", value)

        before = bytes(text[r_off:end])
        staged.append(
            (
                r_off,
                width,
                after,
                Patch(
                    reloc_index=reloc.index,
                    reloc_type=reloc.reloc_type,
                    reloc_name=RELOC_NAMES[reloc.reloc_type],
                    symbol=sym.name,
                    offset=r_off,
                    width=width,
                    s=s_addr,
                    a=addend,
                    p=p_addr,
                    value=value,
                    before_hex=before.hex(),
                    after_hex=after.hex(),
                ),
            )
        )

    # --- patch non-overlap validation (ordered by offset) ---------------
    staged.sort(key=lambda t: (t[0], t[1]))
    for (off1, w1, _, p1), (off2, w2, _, p2) in zip(staged, staged[1:]):
        if off2 < off1 + w1:
            raise AuditError(
                f"补丁写入区间重叠：重定位项[{p1.reloc_index}] "
                f"[{_hex(off1)}, {_hex(off1 + w1)}) 与 重定位项[{p2.reloc_index}] "
                f"[{_hex(off2)}, {_hex(off2 + w2)})",
                position=f"重定位项[{p2.reloc_index}].r_offset",
            )

    # All checks passed: only now mutate the (in-memory) section image.
    patches: list[Patch] = []
    for off, _w, after, patch in staged:
        text[off : off + len(after)] = after
        patches.append(patch)

    original = bytes(obj.text_bytes)
    patched = bytes(text)
    return AuditResult(
        text_size=text_size,
        original_text_sha256=hashlib.sha256(original).hexdigest(),
        patched_text_sha256=hashlib.sha256(patched).hexdigest(),
        file_sha256=hashlib.sha256(data).hexdigest(),
        file_size=len(data),
        load_base=load_base,
        external_symbols=tuple(external_names),
        patches=tuple(patches),
    )
