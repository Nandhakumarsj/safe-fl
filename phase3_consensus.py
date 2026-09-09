import os
import shutil
import warnings

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _has_host_compiler():
    if not torch.cuda.is_available():
        return False
    if shutil.which("ninja") is None:
        return False
    if os.name == "nt":
        return shutil.which("cl") is not None
    return shutil.which("g++") is not None or shutil.which("clang++") is not None


def _load_extension():
    # 1. Check if extension was pre-compiled (e.g. via setup.py or Docker build)
    try:
        import consensus_cuda
        return consensus_cuda
    except ImportError:
        pass

    if not torch.cuda.is_available():
        return None

    # 2. Check if C++ compiler and ninja are in PATH to avoid internal subprocess WinError
    if not _has_host_compiler():
        # Informative note without dumping scary Win32 process tracebacks
        warnings.warn(
            "CUDA C++ compiler (cl/g++) or ninja not found in PATH. "
            "Falling back to PyTorch GPU median consensus. "
            "(To compile custom CUDA kernels, use the provided Docker container or install C++ build tools)."
        )
        return None

    try:
        return load(
            name="consensus_cuda",
            sources=[
                os.path.join(_THIS_DIR, "consensus_kernel.cu"),
                os.path.join(_THIS_DIR, "consensus_binding.cpp"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math", "--allow-unsupported-compiler"],
            verbose=False,
        )
    except Exception as exc:
        warnings.warn(
            "Failed to compile consensus CUDA extension. "
            "Falling back to PyTorch median. Original error: "
            f"{exc}"
        )
        return None


consensus_lib = _load_extension()


class MultiStreamConsensus:
    def __init__(self, num_streams=8, device="cuda"):
        self.num_streams = int(num_streams)
        self.device = device

        self.use_cuda = (
            device.startswith("cuda")
            and torch.cuda.is_available()
            and consensus_lib is not None
        )

        if self.use_cuda:
            self.streams = [
                torch.cuda.Stream(device=device)
                for _ in range(self.num_streams)
            ]
            self.stream_ptrs = [s.cuda_stream for s in self.streams]
        else:
            self.streams = []
            self.stream_ptrs = []

    def aggregate_neighbors(self, neighbor_updates):
        K = len(neighbor_updates)

        if K not in (3, 8):
            raise ValueError(
                f"MultiStreamConsensus only supports K in {{3, 8}}, got K={K}"
            )

        stacked = torch.stack(neighbor_updates, dim=0)

        if not self.use_cuda:
            return torch.median(stacked, dim=0).values

        transposed = stacked.T.contiguous()

        aggregated = consensus_lib.coordinate_wise_median(
            transposed,
            self.stream_ptrs,
        )

        current_stream = torch.cuda.current_stream(device=self.device)

        events = []
        for s in self.streams:
            transposed.record_stream(s)
            aggregated.record_stream(s)

            event = torch.cuda.Event()
            event.record(s)
            events.append(event)

        for event in events:
            current_stream.wait_event(event)

        return aggregated
