"""Common CLI syntax and the usage-error status (distinct from partial=2)."""

import argparse
import json
import sys


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(json.dumps({"status": "usage_error", "certificate_verified": False,
                          "mismatches": [message]}))
        self.exit(3, f"{self.prog}: error: {message}\n")


def bounded_integer(minimum=0, maximum=(1 << 63) - 1):
    def parse(value):
        try:
            result = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("expected an integer") from None
        if not minimum <= result <= maximum:
            raise argparse.ArgumentTypeError(f"integer must be in {minimum}..{maximum}")
        return result
    return parse


def task_range(value):
    bounds = value.split(":")
    if len(bounds) != 2 or not all(b.isascii() and b.isdecimal() for b in bounds):
        raise argparse.ArgumentTypeError("range must be START:STOP with unsigned integers")
    first, last = map(bounded_integer(), bounds)
    if last < first:
        raise argparse.ArgumentTypeError("range STOP must be at least START")
    return first, last
