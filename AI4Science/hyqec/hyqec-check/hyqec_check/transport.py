"""Bounded proof transport; no decompressed record file is ever created."""

from contextlib import contextmanager
import gzip
import struct
import subprocess
import zlib

from .algebra import ValidationError
from .input import strict_json

CHUNK_BYTES = 64 * 1024


def read_exact(stream, count):
    result = stream.read(count)
    if len(result) != count:
        raise ValidationError("truncated certificate container")
    return result


@contextmanager
def open_certificate(path, magic, validate, max_header, max_permutation_work):
    """Keep the validated header and its record stream on the same open file."""
    try:
        with path.open("rb") as source:
            compressed = source.read(2) == b"\x1f\x8b"
            source.seek(0)
            # GzipFile does not own source, so both contexts must be closed.
            with gzip.GzipFile(fileobj=source, mode="rb") if compressed else source as stream:
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


def run_stream(command, stream, max_proof_bytes):
    """Feed an unseekable pipe with backpressure and enforce the record limit.

    Native stdout contains one small verdict. On early native rejection a
    broken pipe stops decoding immediately. Success requires decoder EOF;
    errors (including a late CRC error) kill/reap the child before propagating.
    """
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            total = 0
            reached_eof = False
            try:
                while True:
                    chunk = stream.read(min(CHUNK_BYTES, max_proof_bytes - total + 1))
                    total += len(chunk)
                    if total > max_proof_bytes:
                        raise ValidationError("proof exceeds configured byte limit")
                    if not chunk:
                        reached_eof = True
                        break
                    process.stdin.write(chunk)
                process.stdin.close()
            except BrokenPipeError:
                # The native checker has stopped consuming the proof.
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            process.stdin = None
            stdout, _ = process.communicate()
            if process.returncode == 0 and not reached_eof:
                raise ValidationError("native checker stopped before proof EOF")
            return subprocess.CompletedProcess(command, process.returncode, stdout.decode("utf-8"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
