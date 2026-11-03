"""The exact unsigned trace arithmetic in BNB_CERT_SPEC.md section 8."""

MASK = (1 << 64) - 1
MAX_COUNT = (1 << 63) - 1
BASE = 0x100000001B3


def mix64(value: int) -> int:
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def fingerprint(lab) -> int:
    value = 0
    for q, a in enumerate(lab):
        value ^= mix64(8 * q + a + 1)
    return value


def packed(weight: int, active: int, rule: int, generator: int = 65535, size: int = 0) -> int:
    return weight | (active << 8) | (rule << 24) | (generator << 32) | (size << 48)


def concatenate(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    count = a[0] + b[0]
    if count > MAX_COUNT:
        raise ValueError("node count overflow")
    return count, (a[1] * pow(BASE, 2 * b[0], 1 << 64) + b[1]) & MASK


def entry(lab, weight, active, rule, generator=65535, size=0):
    return 1, (fingerprint(lab) * BASE + packed(weight, active, rule, generator, size)) & MASK
