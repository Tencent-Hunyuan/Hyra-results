"""Small C ABI for conditional graphs; PyTorch keeps the captured tensor pools."""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
import shutil
from pathlib import Path
import subprocess

import torch


def library():
    global _LIBRARY
    if '_LIBRARY' in globals():
        return _LIBRARY
    source = Path(__file__).with_name('conditional.cu')
    major, minor = torch.cuda.get_device_capability()
    identity = source.read_bytes() + f'{major}.{minor}:{torch.version.cuda}'.encode()
    signature = hashlib.sha256(identity).hexdigest()[:16]
    build = Path(os.environ.get('HYFF_CACHE_DIR', Path.home() / '.cache' / 'hyff')) / 'native'
    build.mkdir(parents=True, exist_ok=True)
    output = build / f'conditional_{signature}.so'
    if not output.exists():
        from torch.utils.cpp_extension import CUDA_HOME
        nvcc = shutil.which('nvcc') or (str(Path(CUDA_HOME) / 'bin/nvcc') if CUDA_HOME else None)
        if not nvcc:
            raise RuntimeError('CUDA toolkit nvcc is required for native MD')
        major, minor = torch.cuda.get_device_capability()
        command = [nvcc, '-std=c++17', '-O3', '-shared',
            '-Xcompiler', '-fPIC', f'-arch=sm_{major}{minor}', '--cudart', 'shared',
            str(source), '-o', str(output)]
        completed = subprocess.run(command, text=True, capture_output=True)
        (build / f'{signature}.excluded.log').write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError('Native CUDA helper compilation failed; the compiler output is in the .log file of the HyFF native cache directory')
    lib = ct.CDLL(str(output))
    signatures = {
        'hyff_capture_graph': [ct.c_void_p, ct.POINTER(ct.c_void_p)],
        'hyff_clone_graph': [ct.c_void_p, ct.POINTER(ct.c_void_p)],
        'hyff_destroy_graph': [ct.c_void_p],
        'hyff_make_loop': [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p)],
        'hyff_make_loop_with_rebuild': [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                      ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p)],
        'hyff_launch_loop': [ct.c_void_p, ct.c_void_p, ct.c_int],
        'hyff_destroy_loop': [ct.c_void_p],
    }
    for name, arguments in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, ct.c_int
    lib.hyff_cuda_error.argtypes = [ct.c_int]
    lib.hyff_cuda_error.restype = ct.c_char_p
    _LIBRARY = lib
    return lib


def check(code):
    if code:
        raise RuntimeError(f'CUDA conditional graph error {code}: {library().hyff_cuda_error(code).decode()}')


class CapturedChild:
    def __init__(self, function, stream):
        self.owner = torch.cuda.CUDAGraph()
        # PyTorch 2.7 normally deletes its raw graph after instantiation.
        # Its supported debug mode keeps that graph so we can clone it via
        # the public CUDA runtime API. Keep the owning graph alive for its memory pool.
        self.owner.enable_debug_mode()
        raw, self.handle = ct.c_void_p(), ct.c_void_p()
        lib = library()
        with torch.cuda.graph(self.owner, stream=stream):
            self.output = function()
            check(lib.hyff_capture_graph(stream.cuda_stream, ct.byref(raw)))
        check(lib.hyff_clone_graph(raw, ct.byref(self.handle)))

    def replay(self):
        """Launch the captured child on its own (between loop launches)."""
        self.owner.replay()

    def close(self):
        if self.handle.value:
            check(library().hyff_destroy_graph(self.handle))
            self.handle = ct.c_void_p()
        if self.owner is not None:
            self.owner.reset()
            self.owner = None


class ConditionalLoop:
    def __init__(self, pre, post, flags, ledger, rebuild=None, neighbor_status=None):
        if flags.dtype != torch.int32 or ledger.dtype != torch.int32:
            raise ValueError('CUDA control buffers must be int32')
        if flags.numel() != 6 or ledger.numel() != (16 if rebuild is not None else 11):
            raise ValueError('Unexpected conditional control buffer size')
        self.handle = ct.c_void_p()
        self.pre, self.post, self.flags, self.ledger = pre, post, flags, ledger
        self.rebuild, self.neighbor_status = rebuild, neighbor_status
        if rebuild is None:
            check(library().hyff_make_loop(pre.handle, post.handle,
                flags.data_ptr(), ledger.data_ptr(), ct.byref(self.handle)))
        else:
            if neighbor_status is None or neighbor_status.dtype != torch.int32 or neighbor_status.numel() != 4:
                raise ValueError('A four-int native neighbor status buffer is required')
            check(library().hyff_make_loop_with_rebuild(pre.handle, rebuild.handle, post.handle,
                flags.data_ptr(), ledger.data_ptr(), neighbor_status.data_ptr(), ct.byref(self.handle)))

    def launch(self, count):
        if type(count) is not int or count < 1:
            raise ValueError('A positive integer chunk size is required')
        check(library().hyff_launch_loop(self.handle,
            torch.cuda.current_stream(self.ledger.device).cuda_stream, count))

    def close(self):
        if self.handle.value:
            check(library().hyff_destroy_loop(self.handle))
            self.handle = ct.c_void_p()
