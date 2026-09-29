"""Minimal, strict ELF64 little-endian parser for relocatable objects.

Only the structures needed to audit ``R_X86_64_64`` / ``R_X86_64_PC32``
relocations against a single ``.text`` section are implemented.  Every
table access is bounds-checked and raises :class:`AuditError`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

R_X86_64_64 = 1
R_X86_64_PC32 = 2
SUPPORTED_RELOC_TYPES = (R_X86_64_64, R_X86_64_PC32)

DT_TEXT = ".text"
DT_RELA_TEXT = ".rela.text"

SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4

SHN_UNDEF = 0
SHN_XINDEX = 0xFFFF

EI_NIDENT = 16
ELFMAG = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
ET_REL = 1
EM_X86_64 = 62
EV_CURRENT = 1

EHDR_SIZE = 64
SHDR_SIZE = 64
SYM_SIZE = 24
RELA_SIZE = 24

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
UINT64_MASK = (1 << 64) - 1


class AuditError(ValueError):
    """A validation failure.

    ``position`` identifies the first violating location (byte offset,
    section/table index or relocation entry ordinal) so the page can
    point the reviewer at exactly where the object stopped being valid.
    """

    def __init__(self, message: str, position: str | None = None):
        super().__init__(message)
        self.message = message
        self.position = position

    def with_position(self, position: str) -> "AuditError":
        if self.position is None:
            self.position = position
        return self


class _Buf:
    def __init__(self, data: bytes):
        self.data = data

    def _need(self, off: int, n: int, what: str) -> None:
        if off < 0 or n < 0 or off > len(self.data) - n:
            raise AuditError(f"{what} 越界（offset={off}, size={n}, 文件长度={len(self.data)}）")

    def u8(self, off: int, what: str) -> int:
        self._need(off, 1, what)
        return self.data[off]

    def u16(self, off: int, what: str) -> int:
        self._need(off, 2, what)
        return struct.unpack_from("<H", self.data, off)[0]

    def u32(self, off: int, what: str) -> int:
        self._need(off, 4, what)
        return struct.unpack_from("<I", self.data, off)[0]

    def u64(self, off: int, what: str) -> int:
        self._need(off, 8, what)
        return struct.unpack_from("<Q", self.data, off)[0]

    def i64(self, off: int, what: str) -> int:
        self._need(off, 8, what)
        return struct.unpack_from("<q", self.data, off)[0]

    def slice(self, off: int, n: int, what: str) -> bytes:
        self._need(off, n, what)
        return self.data[off : off + n]


@dataclass(frozen=True)
class Section:
    index: int
    name: str
    sh_type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    addralign: int
    entsize: int

    def bytes(self, buf: _Buf) -> bytes:
        return buf.slice(self.offset, self.size, f"节 {self.index}({self.name}) 数据")


@dataclass(frozen=True)
class Symbol:
    index: int
    name: str
    st_info: int
    st_other: int
    shndx: int
    value: int
    size: int

    @property
    def is_undefined(self) -> bool:
        return self.shndx == SHN_UNDEF


@dataclass(frozen=True)
class Relocation:
    index: int  # 0-based ordinal within the RELA section
    offset: int  # r_offset within .text
    sym_index: int  # r_info >> 32
    reloc_type: int  # r_info & 0xffffffff
    addend: int  # signed r_addend
    symbol: Symbol | None = None


@dataclass(frozen=True)
class ElfObject:
    sections: list[Section]
    text: Section
    rela: Section
    symtab: Section
    strtab: Section
    text_bytes: bytes
    symbols: list[Symbol]
    relocations: list[Relocation]


def _cstring(buf: bytes, off: int, what: str) -> str:
    if off < 0 or off >= len(buf):
        raise AuditError(f"{what} 的字符串偏移 {off} 越界")
    end = buf.find(b"\x00", off)
    if end < 0:
        raise AuditError(f"{what} 的字符串未以 NUL 结尾（偏移 {off}）")
    raw = buf[off:end]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_ehdr(buf: _Buf) -> tuple[int, int, int, int]:
    """Validate e_ident + fixed header; return (shoff, shentsize, shnum, shstrndx)."""
    buf._need(0, EI_NIDENT, "ELF 标识")
    if buf.data[:4] != ELFMAG:
        raise AuditError("非法 ELF 魔数", position="e_ident[0..3]")
    ei_class = buf.u8(4, "EI_CLASS")
    if ei_class != ELFCLASS64:
        raise AuditError(f"仅接受 ELF64（EI_CLASS={ei_class}）", position="EI_CLASS")
    ei_data = buf.u8(5, "EI_DATA")
    if ei_data != ELFDATA2LSB:
        raise AuditError(f"仅接受小端序（EI_DATA={ei_data}）", position="EI_DATA")
    ei_version = buf.u8(6, "EI_VERSION")
    if ei_version != EV_CURRENT:
        raise AuditError(f"非法 EI_VERSION={ei_version}", position="EI_VERSION")
    # Padding bytes EI_OSABI etc. are not constrained.

    buf._need(0, EHDR_SIZE, "ELF 头")
    e_type = buf.u16(16, "e_type")
    if e_type != ET_REL:
        raise AuditError(f"仅接受 ET_REL（e_type={e_type}）", position="e_type")
    e_machine = buf.u16(18, "e_machine")
    if e_machine != EM_X86_64:
        raise AuditError(f"仅接受 EM_X86_64（e_machine={e_machine}）", position="e_machine")
    e_version = buf.u32(20, "e_version")
    if e_version != EV_CURRENT:
        raise AuditError(f"非法 e_version={e_version}", position="e_version")

    shoff = buf.u64(40, "e_shoff")
    phoff = buf.u64(32, "e_phoff")
    phentsize = buf.u16(54, "e_phentsize")
    phnum = buf.u16(56, "e_phnum")
    shentsize = buf.u16(58, "e_shentsize")
    shnum = buf.u16(60, "e_shnum")
    shstrndx = buf.u16(62, "e_shstrndx")

    if phoff != 0 or phentsize != 0 or phnum != 0:
        raise AuditError(
            "ET_REL 文件不得包含程序头表（e_phoff/e_phnum 必须为 0）",
            position="e_phoff",
        )
    if shoff == 0:
        raise AuditError("缺少节头表（e_shoff=0）", position="e_shoff")
    if shentsize != SHDR_SIZE:
        raise AuditError(f"非法 e_shentsize={shentsize}（应为 {SHDR_SIZE}）", position="e_shentsize")
    if shnum == 0 or shstrndx == SHN_XINDEX:
        # Extended numbering (SHN_XINDEX / e_shnum==0) is rejected: the
        # auditor only deals with ordinary, well-formed relocatable objects.
        raise AuditError("不支持扩展节索引编号（e_shnum=0 / SHN_XINDEX）", position="e_shnum/e_shstrndx")
    if shstrndx >= shnum:
        raise AuditError(f"e_shstrndx={shstrndx} 超出节表范围 0..{shnum - 1}", position="e_shstrndx")
    # Section table itself must fit the file.
    buf._need(shoff, shentsize * shnum, "节头表")
    return shoff, shentsize, shnum, shstrndx


def _read_sections(buf: _Buf, shoff: int, shnum: int, shstrndx: int) -> list[Section]:
    raw: list[tuple] = []
    for i in range(shnum):
        base = shoff + i * SHDR_SIZE
        name_off = buf.u32(base + 0, f"节[{i}].sh_name")
        sh_type = buf.u32(base + 4, f"节[{i}].sh_type")
        flags = buf.u64(base + 8, f"节[{i}].sh_flags")
        addr = buf.u64(base + 16, f"节[{i}].sh_addr")
        offset = buf.u64(base + 24, f"节[{i}].sh_offset")
        size = buf.u64(base + 32, f"节[{i}].sh_size")
        link = buf.u32(base + 40, f"节[{i}].sh_link")
        info = buf.u32(base + 44, f"节[{i}].sh_info")
        addralign = buf.u64(base + 48, f"节[{i}].sh_addralign")
        entsize = buf.u64(base + 56, f"节[{i}].sh_entsize")
        raw.append((name_off, sh_type, flags, addr, offset, size, link, info, addralign, entsize))

    # Section name string table.
    shstr_raw = raw[shstrndx]
    if shstr_raw[1] != SHT_STRTAB:
        raise AuditError(
            f"e_shstrndx 指向的节[{shstrndx}]类型为 {shstr_raw[1]}，不是 STRTAB",
            position=f"节[{shstrndx}].sh_type",
        )
    shstr_off, shstr_size = shstr_raw[4], shstr_raw[5]
    shstrtab = buf.slice(shstr_off, shstr_size, "节名字符串表(.shstrtab)")

    sections: list[Section] = []
    for i, t in enumerate(raw):
        if i == 0:
            if t[0] != 0:
                raise AuditError("保留节 0 的 sh_name 必须为 0", position="节[0].sh_name")
            if t[1] != 0:
                raise AuditError("保留节 0 的 sh_type 必须为 SHT_NULL",
                                 position="节[0].sh_type")
            if any(t[j] != 0 for j in (2, 3, 4, 5, 6, 7, 8, 9)):
                raise AuditError("保留节 0 的字段必须全部为 0", position="节[0]")
            name = ""
        else:
            name = _cstring(shstrtab, t[0], f"节[{i}].sh_name")
        sections.append(
            Section(
                index=i,
                name=name,
                sh_type=t[1],
                flags=t[2],
                addr=t[3],
                offset=t[4],
                size=t[5],
                link=t[6],
                info=t[7],
                addralign=t[8],
                entsize=t[9],
            )
        )

    # Every non-NOBITS section's bytes must live inside the file.  We do
    # not accept SHT_NOBITS in audited objects beyond ignoring index 0.
    for sec in sections:
        if sec.index == 0:
            continue
        if sec.sh_type == 8:  # SHT_NOBITS
            if sec.offset != 0:
                raise AuditError(f"节[{sec.index}]({sec.name}) 为 NOBITS 但 sh_offset!=0",
                                 position=f"节[{sec.index}].sh_offset")
            continue
        if sec.size > 0:
            buf._need(sec.offset, sec.size, f"节[{sec.index}]({sec.name}) 数据")
    return sections


def _read_symbols(buf: _Buf, sections: list[Section], symtab: Section) -> list[Symbol]:
    if symtab.entsize not in (0, SYM_SIZE):
        raise AuditError(
            f"符号表 entsize={symtab.entsize} 非法（应为 {SYM_SIZE}）",
            position=f"节[{symtab.index}].sh_entsize",
        )
    strtab_idx = symtab.link
    if not (0 < strtab_idx < len(sections)):
        raise AuditError(
            f"符号表 sh_link={strtab_idx} 越界", position=f"节[{symtab.index}].sh_link"
        )
    strtab = sections[strtab_idx]
    if strtab.sh_type != SHT_STRTAB:
        raise AuditError(
            f"符号表 sh_link 指向节[{strtab_idx}]，类型为 {strtab.sh_type}，不是 STRTAB",
            position=f"节[{strtab_idx}].sh_type",
        )
    strtab_bytes = strtab.bytes(buf)
    data = symtab.bytes(buf)
    if symtab.size == 0 or len(data) % SYM_SIZE != 0:
        raise AuditError(
            f"符号表大小 {symtab.size} 不是 {SYM_SIZE} 的整数倍（表已截断）",
            position=f"节[{symtab.index}].sh_size",
        )
    count = len(data) // SYM_SIZE
    symbols: list[Symbol] = []
    for i in range(count):
        base = i * SYM_SIZE
        st_name = struct.unpack_from("<I", data, base)[0]
        st_info = data[base + 4]
        st_other = data[base + 5]
        st_shndx = struct.unpack_from("<H", data, base + 6)[0]
        st_value = struct.unpack_from("<Q", data, base + 8)[0]
        st_size = struct.unpack_from("<Q", data, base + 16)[0]
        if i == 0:
            if st_name != 0 or st_info != 0 or st_other != 0 or st_shndx != 0 or st_value != 0 or st_size != 0:
                raise AuditError("符号表第 0 项必须为全零保留项", position="符号[0]")
            name = ""
        else:
            name = _cstring(strtab_bytes, st_name, f"符号[{i}].st_name")
        symbols.append(Symbol(i, name, st_info, st_other, st_shndx, st_value, st_size))
    return symbols


def _read_relocations(buf: _Buf, rela: Section, sections: list[Section]) -> list[Relocation]:
    if rela.entsize not in (0, RELA_SIZE):
        raise AuditError(
            f"RELA 节 entsize={rela.entsize} 非法（应为 {RELA_SIZE}）",
            position=f"节[{rela.index}].sh_entsize",
        )
    data = rela.bytes(buf)
    if len(data) % RELA_SIZE != 0:
        raise AuditError(
            f"RELA 节大小 {rela.size} 不是 {RELA_SIZE} 的整数倍（表已截断）",
            position=f"节[{rela.index}].sh_size",
        )
    relocs: list[Relocation] = []
    for i in range(len(data) // RELA_SIZE):
        base = i * RELA_SIZE
        r_offset = struct.unpack_from("<Q", data, base)[0]
        r_info = struct.unpack_from("<Q", data, base + 8)[0]
        r_addend = struct.unpack_from("<q", data, base + 16)[0]
        r_sym = r_info >> 32
        r_type = r_info & 0xFFFFFFFF
        relocs.append(Relocation(i, r_offset, r_sym, r_type, r_addend))
    return relocs


def parse_object(data: bytes) -> ElfObject:
    """Parse and structurally validate an ELF64 LE ET_REL object's tables."""
    if not isinstance(data, (bytes, bytearray)):
        raise AuditError("文件内容必须为字节序列")
    buf = _Buf(bytes(data))
    shoff, shentsize, shnum, shstrndx = _read_ehdr(buf)
    sections = _read_sections(buf, shoff, shnum, shstrndx)

    text_sections = [s for s in sections if s.name == DT_TEXT]
    if len(text_sections) != 1:
        raise AuditError(
            f"必须恰好存在一个名为 {DT_TEXT} 的节（找到 {len(text_sections)} 个）",
            position="节名字符串表",
        )
    text = text_sections[0]
    if text.sh_type != SHT_PROGBITS:
        raise AuditError(
            f"{DT_TEXT} 节类型为 {text.sh_type}，不是 PROGBITS",
            position=f"节[{text.index}].sh_type",
        )

    rela_sections = [s for s in sections if s.sh_type == SHT_RELA]
    if len(rela_sections) != 1:
        raise AuditError(
            f"必须恰好存在一个 RELA 节（找到 {len(rela_sections)} 个）",
            position="节头表",
        )
    rela = rela_sections[0]
    if rela.info != text.index:
        raise AuditError(
            f"RELA 节[{rela.index}] 的 sh_info={rela.info} 未指向 {DT_TEXT}"
            f"（节索引 {text.index}）",
            position=f"节[{rela.index}].sh_info",
        )

    sym_idx = rela.link
    if not (0 < sym_idx < len(sections)):
        raise AuditError(
            f"RELA 节 sh_link={sym_idx} 越界", position=f"节[{rela.index}].sh_link"
        )
    symtab = sections[sym_idx]
    if symtab.sh_type != SHT_SYMTAB:
        raise AuditError(
            f"RELA 节 sh_link 指向节[{sym_idx}]，类型为 {symtab.sh_type}，不是 SYMTAB",
            position=f"节[{sym_idx}].sh_type",
        )

    symbols = _read_symbols(buf, sections, symtab)
    strtab = sections[symtab.link]
    relocations = _read_relocations(buf, rela, sections)

    return ElfObject(
        sections=sections,
        text=text,
        rela=rela,
        symtab=symtab,
        strtab=strtab,
        text_bytes=text.bytes(buf),
        symbols=symbols,
        relocations=relocations,
    )
