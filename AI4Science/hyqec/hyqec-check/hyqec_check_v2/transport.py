"""Bounded single-member v2 gzip decoder; v1 transport remains unchanged."""
from contextlib import contextmanager
import gzip
import struct
import zlib

from hyqec_check.algebra import ValidationError
from hyqec_check.input import strict_json
from hyqec_check.transport import CHUNK_BYTES, read_exact


def container_version(path):
    with path.open("rb") as source:
        magic = source.read(8)
        if magic[:2] == b"\x1f\x8b":
            source.seek(0)
            try:
                with gzip.GzipFile(fileobj=source, mode="rb") as stream:
                    magic = stream.read(8)
            except (zlib.error, EOFError) as error:
                raise ValidationError(f"invalid gzip stream: {error}") from error
        if magic == b"SBCOV1\r\n":
            return 1
        if magic == b"SBCOV2\r\n":
            return 2
        raise ValidationError("invalid coverage magic")


class SingleGzip:
    def __init__(self, source):
        self.source = source
        self.decoder = zlib.decompressobj(wbits=31)
        self.pending = b""
        self.finished = False
        # zlib validates the gzip header and CRC/ISIZE. Draft-2 requires
        # single-member EOF, but does not require checking mtime/name/OS.

    def read(self, count):
        if count < 0:
            raise ValueError("unbounded gzip read is forbidden")
        output = bytearray()
        while len(output) < count and not self.finished:
            if not self.pending:
                self.pending = self.source.read(CHUNK_BYTES)
                if not self.pending:
                    raise ValidationError("truncated gzip member")
            chunk = self.decoder.decompress(self.pending, count - len(output))
            output.extend(chunk)
            self.pending = self.decoder.unconsumed_tail
            if self.decoder.eof:
                if self.decoder.unused_data or self.source.read(1):
                    raise ValidationError("data follows v2 gzip member")
                self.finished = True
        return bytes(output)


@contextmanager
def open_certificate(path, magic, validate, max_header, max_permutation_work):
    try:
        with path.open("rb") as source:
            compressed = source.read(2) == b"\x1f\x8b"
            source.seek(0)
            stream = SingleGzip(source) if compressed else source
            if read_exact(stream, 8) != magic:
                raise ValidationError("invalid coverage magic")
            length, = struct.unpack("<Q", read_exact(stream, 8))
            if length > max_header:
                raise ValidationError("header exceeds configured byte limit")
            header = strict_json(read_exact(stream, length))
            instance = validate(header, max_permutation_work=max_permutation_work)
            yield header, instance, stream
    except (zlib.error, EOFError) as error:
        raise ValidationError(f"invalid gzip stream: {error}") from error
