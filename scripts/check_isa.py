#!/usr/bin/env python3
"""Assert an ARM ELF declares the CPU architecture baseline it claims.

    scripts/check_isa.py <binary> <expected>   # v6 | v7 | aarch64 | x86_64

The failure this prevents is quiet. An ARMv7 binary installs cleanly on a Pi Zero
and dies with SIGILL at the first ARMv7-only instruction, which may be inside a
code path that does not run until a query arrives. A triple's *name* is not
evidence that the compiler emitted that baseline; the ELF attributes are.

# Why this parses the section instead of calling readelf

`readelf -A` renders Tag_CPU_arch differently depending on which readelf ran: GNU
binutils prints `Tag_CPU_arch: v6`, LLVM prints `Description: ARM v6`. A grep
tuned to one silently *passes* against the other — it finds no match, and a
`grep -q` in an `if` reads that as "attribute absent" or, worse, the pipeline is
written so the absence looks like success. A check that can pass by not working is
worse than no check.

The numeric encoding, by contrast, is fixed by the ARM ABI (IHI 0045,
"Addenda to ABI for the ARM Architecture"): the build-attributes section is a
sequence of ULEB128 tag/value pairs, Tag_CPU_arch is tag 6, and its value is 6 for
ARMv6 and 10 for ARMv7. That contract does not vary by toolchain or version.
"""

from __future__ import annotations

import struct
import sys

# Tag_CPU_arch values, from the ARM ABI addenda. Only the ones this project ships
# are listed: a value outside this map is reported by number rather than guessed
# at, because naming an architecture wrongly here would be worse than not naming
# it.
CPU_ARCH = {
    1: "v4",
    2: "v4T",
    3: "v5T",
    4: "v5TE",
    5: "v5TEJ",
    6: "v6",
    7: "v6KZ",
    8: "v6T2",
    9: "v6K",
    10: "v7",
    11: "v6-M",
    12: "v6S-M",
    13: "v7E-M",
    14: "v8-A",
}

TAG_CPU_NAME = 5
TAG_CPU_ARCH = 6
TAG_FP_ARCH = 10

EM_ARM = 40
EM_AARCH64 = 183
EM_X86_64 = 62


def uleb128(buf: bytes, at: int) -> tuple[int, int]:
    """Decode a ULEB128 at `at`, returning (value, next offset)."""
    result = 0
    shift = 0
    while True:
        byte = buf[at]
        at += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, at
        shift += 7


def sections(blob: bytes) -> tuple[int, dict[str, bytes]]:
    """Return (e_machine, {section name: contents}) for a 32- or 64-bit ELF."""
    if blob[:4] != b"\x7fELF":
        raise SystemExit("not an ELF file")
    bits = 32 if blob[4] == 1 else 64
    machine = struct.unpack_from("<H", blob, 18)[0]

    if bits == 32:
        e_shoff = struct.unpack_from("<I", blob, 32)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 46)
        name_off, off_off, size_off = 0, 16, 20
        word = "<I"
    else:
        e_shoff = struct.unpack_from("<Q", blob, 40)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 58)
        name_off, off_off, size_off = 0, 24, 32
        word = "<Q"

    def header(i: int) -> tuple[int, int, int]:
        base = e_shoff + i * e_shentsize
        name = struct.unpack_from("<I", blob, base + name_off)[0]
        offset = struct.unpack_from(word, blob, base + off_off)[0]
        size = struct.unpack_from(word, blob, base + size_off)[0]
        return name, offset, size

    _, str_off, str_size = header(e_shstrndx)
    strtab = blob[str_off : str_off + str_size]

    out: dict[str, bytes] = {}
    for i in range(e_shnum):
        name_idx, offset, size = header(i)
        end = strtab.find(b"\x00", name_idx)
        name = strtab[name_idx:end].decode("utf-8", "replace")
        out[name] = blob[offset : offset + size]
    return machine, out


def arm_attributes(section: bytes) -> dict[int, int | str]:
    """Parse the aeabi file-attribute subsection into {tag: value}."""
    if not section or section[0] != ord("A"):
        raise SystemExit("unrecognised .ARM.attributes format version")

    # Tags carrying a NUL-terminated string rather than a ULEB128:
    # Tag_CPU_raw_name(4), Tag_CPU_name(5), Tag_compatibility(32),
    # Tag_also_compatible_with(65), Tag_conformance(67).
    string_tags = {4, 5, 32, 65, 67}

    at = 1
    while at + 4 <= len(section):
        (length,) = struct.unpack_from("<I", section, at)
        if length == 0:
            break
        # The length field covers the subsection *including* itself, starting at
        # `at`.
        subsection_end = min(at + length, len(section))
        vendor_end = section.find(b"\x00", at + 4)
        vendor = section[at + 4 : vendor_end].decode("ascii", "replace")
        cursor = vendor_end + 1
        if vendor != "aeabi":
            at = subsection_end
            continue

        attrs: dict[int, int | str] = {}
        while cursor < subsection_end:
            scope_start = cursor
            scope, cursor = uleb128(section, cursor)
            (scope_size,) = struct.unpack_from("<I", section, cursor)
            # `scope_size` is measured from the scope tag byte, not from after the
            # size word. Getting this wrong overruns by exactly 4 bytes and the
            # decoder walks off the end of the section — which is how this was
            # caught, since an ARM attributes section is usually the last thing in
            # the file and there is nothing after it to misparse.
            scope_end = min(scope_start + scope_size, subsection_end, len(section))
            cursor += 4

            if scope != 1:  # not Tag_File: per-section or per-symbol scope.
                at = scope_end
                cursor = scope_end
                continue

            while cursor < scope_end:
                tag, cursor = uleb128(section, cursor)
                if cursor >= scope_end:
                    break
                if tag in string_tags:
                    stop = section.find(b"\x00", cursor)
                    if stop < 0 or stop > scope_end:
                        break
                    attrs[tag] = section[cursor:stop].decode("ascii", "replace")
                    cursor = stop + 1
                else:
                    value, cursor = uleb128(section, cursor)
                    attrs[tag] = value
            return attrs
        at = subsection_end
    raise SystemExit("no aeabi file attributes in .ARM.attributes")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    path, expected = argv[1], argv[2]
    blob = open(path, "rb").read()
    machine, secs = sections(blob)

    # Architectures with no build-attributes section: the machine type is the whole
    # check, and it is sufficient — an ELF cannot claim two of these at once.
    by_machine = {"aarch64": EM_AARCH64, "x86_64": EM_X86_64}
    if expected in by_machine:
        want_machine = by_machine[expected]
        if machine != want_machine:
            print(f"FAIL {path}: e_machine {machine}, expected {want_machine} ({expected})")
            return 1
        print(f"ok   {path}: {expected} (e_machine {machine})")
        return 0

    if machine != EM_ARM:
        print(f"FAIL {path}: e_machine {machine}, expected {EM_ARM} (ARM)")
        return 1

    section = secs.get(".ARM.attributes")
    if not section:
        print(f"FAIL {path}: no .ARM.attributes section, so no baseline is declared")
        return 1

    attrs = arm_attributes(section)
    arch = attrs.get(TAG_CPU_ARCH)
    name = CPU_ARCH.get(arch, f"unknown({arch})") if isinstance(arch, int) else "absent"
    cpu = attrs.get(TAG_CPU_NAME, "?")
    fp = attrs.get(TAG_FP_ARCH, "?")
    print(f"     {path}: Tag_CPU_arch={arch} ({name}) Tag_CPU_name={cpu} Tag_FP_arch={fp}")

    want = {v: k for k, v in CPU_ARCH.items()}.get(expected)
    if want is None:
        print(f"FAIL unknown expected baseline '{expected}'")
        return 2
    if arch != want:
        print(f"FAIL {path}: declares {name}, expected {expected}")
        print("     a binary built for a newer ISA installs cleanly and then")
        print("     dies with SIGILL when a query reaches an unsupported instruction")
        return 1
    print(f"ok   {path}: {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))