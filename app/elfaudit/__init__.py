"""ELF64 LE ET_REL relocation auditor (R_X86_64_64 / R_X86_64_PC32)."""

from .audit import AuditError, audit_object
from .elf import (
    DT_TEXT,
    R_X86_64_64,
    R_X86_64_PC32,
)

__all__ = [
    "AuditError",
    "audit_object",
    "R_X86_64_64",
    "R_X86_64_PC32",
    "DT_TEXT",
]
