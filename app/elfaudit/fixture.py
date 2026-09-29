"""Synthetic ELF64 LE ET_REL fixture builder (no binutils required).

The layout mirrors what ``gcc -c -fno-asynchronous-unwind-tables`` would
emit for a tiny object::

    [0] NULL
    [1] .text       PROGBITS ALLOC+EXECINSTR
    [2] .rela.text  RELA     (sh_info -> .text, sh_link -> .symtab)
    [3] .symtab     SYMTAB   (sh_link -> .strtab)
    [4] .strtab     STRTAB
    [5] .shstrtab   STRTAB

Optional malformed variants let tests exercise every rejection path.
"""

from __future__ import annotations

import struct

ET_REL = 1
EM_X86_64 = 62
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_REL = 9
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
SHF_INFO_LINK = 0x40
R_X86_64_64 = 1
R_X86_64_PC32 = 2

EHDR_SIZE = 64
SHDR_SIZE = 64
SYM_SIZE = 24
RELA_SIZE = 24
REL_SIZE = 16


def _strtab(strings: list[str]) -> bytes:
    out = bytearray(b"\x00")
    offsets: dict[str, int] = {}
    for s in strings:
        offsets[s] = len(out)
        out += s.encode("utf-8") + b"\x00"
    return bytes(out), offsets


def build_object(
    text: bytes = b"",
    relocs: list[tuple] | None = None,
    externals: list[str] | None = None,
    *,
    e_type: int = ET_REL,
    e_machine: int = EM_X86_64,
    ei_class: int = 2,
    ei_data: int = 1,  # 1 = little endian
    rela_sh_info: int | None = None,
    rela_sh_link: int | None = None,
    symtab_sh_link: int | None = None,
    duplicate_external: str | None = None,
    extra_text: bool = False,
    extra_rela: bool = False,
    with_rel: bool = False,
    defined_symbol: str | None = None,
    defined_in_text: bool = True,
) -> bytes:
    """Build a relocatable object.

    Each reloc is ``(r_offset, r_type, sym, r_addend)`` where ``sym`` is
    an external name (resolved via the symbol table) or a raw integer
    symbol index.
    """
    relocs = relocs or []
    externals = list(externals or [])
    if duplicate_external is not None:
        externals = externals + [duplicate_external]

    names = list(externals)
    if defined_symbol is not None and defined_symbol not in names:
        names.append(defined_symbol)
    strtab, name_off = _strtab(names)

    # ---- symbol table: [0] NULL, one entry per external (SHN_UNDEF) ----
    symtab = bytearray(SYM_SIZE)  # all-zero reserved entry
    sym_index: dict[str, int] = {}
    for i, nm in enumerate(externals, start=1):
        sym_index[nm] = i
        shndx = 0
        value = 0
        if defined_symbol is not None and nm == defined_symbol and defined_in_text:
            shndx = 1  # .text
            value = 0x10
        symtab += struct.pack(
            "<IBBHQQ", name_off[nm], 0x11, 0, shndx, value, 0
        )
    if defined_symbol is not None and defined_symbol not in externals:
        idx = len(symtab) // SYM_SIZE
        sym_index[defined_symbol] = idx
        shndx = 1 if defined_in_text else 0
        value = 0x10 if defined_in_text else 0
        symtab += struct.pack(
            "<IBBHQQ", name_off[defined_symbol], 0x11, 0, shndx, value, 0
        )

    # ---- .rela.text ------------------------------------------------------
    rela = bytearray()
    for r_off, r_type, sym, addend in relocs:
        if isinstance(sym, str):
            if sym not in sym_index:
                raise ValueError(f"重定位引用了未声明的符号 {sym!r}")
            r_sym = sym_index[sym]
        else:
            r_sym = int(sym)
        r_info = (r_sym << 32) | (r_type & 0xFFFFFFFF)
        rela += struct.pack("<QQq", r_off, r_info, addend)

    # ---- section names ---------------------------------------------------
    sec_names = [".text", ".rela.text", ".symtab", ".strtab", ".shstrtab"]
    if extra_text:
        sec_names.append(".text")
    if extra_rela:
        sec_names += [".data", ".rela.data"]
    if with_rel:
        sec_names.append(".rel.data")
    shstrtab, sh_name_off = _strtab(sec_names)

    # Section descriptors: (name, type, flags, addr, link, info, align,
    # entsize, data). Offsets/sizes assigned during layout.
    text_idx = 1
    rela_idx = 2
    symtab_idx = 3
    strtab_idx = 4
    shstrtab_idx = 5

    descriptors = [
        None,  # NULL section
        (".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0, 0, 0, 1, 0, text),
        (
            ".rela.text",
            SHT_RELA,
            SHF_INFO_LINK,
            0,
            rela_sh_link if rela_sh_link is not None else symtab_idx,
            rela_sh_info if rela_sh_info is not None else text_idx,
            8,
            RELA_SIZE,
            bytes(rela),
        ),
        (
            ".symtab",
            SHT_SYMTAB,
            0,
            0,
            symtab_sh_link if symtab_sh_link is not None else strtab_idx,
            1,
            8,
            SYM_SIZE,
            bytes(symtab),
        ),
        (".strtab", SHT_STRTAB, 0, 0, 0, 0, 1, 0, strtab),
        (".shstrtab", SHT_STRTAB, 0, 0, 0, 0, 1, 0, shstrtab),
    ]
    if extra_text:
        descriptors.append(
            (".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0, 0, 0, 1, 0, b"\xc3")
        )
    if extra_rela:
        data_idx = len(descriptors)
        rela_data_idx = data_idx + 1
        descriptors.append(
            (".data", SHT_PROGBITS, SHF_ALLOC, 0, 0, 0, 1, 0, b"\x00" * 8)
        )
        descriptors.append(
            (".rela.data", SHT_RELA, SHF_INFO_LINK, 0, symtab_idx, data_idx, 8, RELA_SIZE, b"")
        )
    if with_rel:
        # SHT_REL carries no addends and must be rejected outright.
        descriptors.append(
            (".rel.data", SHT_REL, 0, 0, symtab_idx, 1, 8, REL_SIZE, b"\x00" * REL_SIZE)
        )

    shnum = len(descriptors)
    shstrndx = shstrtab_idx

    # ---- layout ----------------------------------------------------------
    file_size = EHDR_SIZE
    laid_out = []
    for desc in descriptors[1:]:
        name, stype, flags, addr, link, info, align, entsize, payload = desc
        if align > 1:
            file_size = (file_size + align - 1) // align * align
        offset = file_size
        file_size += len(payload)
        laid_out.append((name, stype, flags, addr, offset, len(payload), link, info, align, entsize, payload))

    shoff = file_size
    file_size += SHDR_SIZE * shnum
    blob = bytearray(file_size)

    endian = "<" if ei_data == 1 else ">"
    # ---- ELF header ------------------------------------------------------
    blob[0:4] = b"\x7fELF"
    blob[4] = ei_class
    blob[5] = ei_data
    blob[6] = 1  # EI_VERSION
    struct.pack_into(endian + "HHI", blob, 16, e_type, e_machine, 1)
    struct.pack_into(endian + "Q", blob, 40, shoff)
    # e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx
    struct.pack_into(endian + "HHHHHH", blob, 52,
                     EHDR_SIZE, 0, 0, SHDR_SIZE, shnum, shstrndx)

    # ---- section payloads ------------------------------------------------
    for (
        name, stype, flags, addr, offset, size, link, info, align, entsize, payload
    ) in laid_out:
        blob[offset : offset + size] = payload

    # ---- section headers -------------------------------------------------
    def put_shdr(idx, name, stype, flags, addr, offset, size, link, info, align, entsize):
        base = shoff + idx * SHDR_SIZE
        struct.pack_into(
            endian + "IIQQQQIIQQ",
            blob,
            base,
            sh_name_off.get(name, 0) if name else 0,
            stype,
            flags,
            addr,
            offset,
            size,
            link,
            info,
            align,
            entsize,
        )

    put_shdr(0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for idx, d in enumerate(laid_out, start=1):
        put_shdr(idx, *d[:-1])

    return bytes(blob)
