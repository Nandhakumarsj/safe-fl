import os
import math
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


def _load_ring_spmv():
    # 1. Check if extension was pre-compiled (e.g. via setup.py or Docker build)
    try:
        import ring_spmv_cuda
        return ring_spmv_cuda
    except ImportError:
        pass

    if not torch.cuda.is_available():
        return None

    # 2. Check if C++ compiler and ninja are in PATH to avoid internal subprocess WinError
    if not _has_host_compiler():
        warnings.warn(
            "CUDA C++ compiler (cl/g++) or ninja not found in PATH. "
            "Falling back to PyTorch GPU ring averaging consensus. "
            "(To compile custom CUDA kernels, use the provided Docker container or install C++ build tools)."
        )
        return None

    try:
        return load(
            name="ring_spmv_cuda",
            sources=[
                os.path.join(_THIS_DIR, "ring_spmv_kernel.cu"),
                os.path.join(_THIS_DIR, "ring_spmv_binding.cpp"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math", "--allow-unsupported-compiler"],
            verbose=False,
        )
    except Exception as exc:
        warnings.warn(
            "Failed to compile ring_spmv CUDA extension. "
            "Falling back to PyTorch ring averaging. Original error: "
            f"{exc}"
        )
        return None


ring_spmv_lib = _load_ring_spmv()


class RingSpMVConsensus:
    def __init__(self, num_streams=8, device="cuda"):
        self.num_streams = int(num_streams)
        self.device = device

        self.use_cuda = (
            device.startswith("cuda")
            and torch.cuda.is_available()
            and ring_spmv_lib is not None
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

    def _cpu_step(self, peer_vectors):
        N = len(peer_vectors)
        out = []

        for i in range(N):
            left = peer_vectors[(i - 1) % N]
            self_vec = peer_vectors[i]
            right = peer_vectors[(i + 1) % N]
            out.append((left + self_vec + right) / 3.0)

        return out

    def step(self, peer_vectors):
        N = len(peer_vectors)

        if N < 3:
            raise ValueError(f"ring topology requires at least 3 peers, got {N}")

        if not self.use_cuda:
            return self._cpu_step(peer_vectors)

        X = torch.stack(peer_vectors, dim=0).contiguous()
        X_new = ring_spmv_lib.ring_spmv(X, self.stream_ptrs)

        current_stream = torch.cuda.current_stream(device=self.device)

        events = []
        for s in self.streams:
            X.record_stream(s)
            X_new.record_stream(s)

            event = torch.cuda.Event()
            event.record(s)
            events.append(event)

        for event in events:
            current_stream.wait_event(event)

        return [X_new[i] for i in range(N)]

    def run_until_converged(self, peer_vectors, eps=1e-2, max_rounds=200):
        X = peer_vectors

        for t in range(1, int(max_rounds) + 1):
            X = self.step(X)

            stacked = torch.stack(X, dim=0)
            spread = (
                stacked.max(dim=0).values - stacked.min(dim=0).values
            ).max().item()

            if spread < eps:
                return X, t

        return X, max_rounds


def parallel_tree_reduce(values, trace=False):
    v = list(values.tolist() if torch.is_tensor(values) else values)
    round_no = 0

    if trace:
        print(f"Round {round_no} (inputs, {len(v)} processors active):")
        print([f"{x:+.4f}" for x in v])

    while len(v) > 1:
        round_no += 1
        next_v = []
        pairs = []

        i = 0
        while i < len(v):
            if i + 1 < len(v):
                s = v[i] + v[i + 1]
                pairs.append((v[i], v[i + 1], s))
                next_v.append(s)
            else:
                next_v.append(v[i])
            i += 2

        if trace:
            active = len(pairs)
            suffix = "s" if active != 1 else ""
            print(f"Round {round_no} ({active} processor{suffix} active):")
            for k, (a, b, s) in enumerate(pairs):
                print(f"    P{k}: {a:+.4f}+{b:+.4f}={s:+.4f}")

        v = next_v

    return v[0], round_no


def parallel_mean_and_std(values, trace=False):
    v = values.tolist() if torch.is_tensor(values) else list(values)
    n = len(v)

    if n == 0:
        raise ValueError("values must not be empty")

    if trace:
        print("--- Branch A: Sum(v_i) -> mean ---")

    total, rounds_a = parallel_tree_reduce(v, trace=trace)
    mean = total / n

    if trace:
        print(f"Sum = {total:+.6f} -> Mean = {mean:+.6f} ({rounds_a} rounds)")
        print("--- Branch B: Sum(v_i^2) -> variance/std ---")

    sq = [x * x for x in v]
    sumsq, rounds_b = parallel_tree_reduce(sq, trace=trace)

    variance = sumsq / n - mean * mean
    std = math.sqrt(max(variance, 0.0))

    if trace:
        print(f"Sum(v^2) = {sumsq:+.6f} -> Variance = {variance:+.6f} -> Std = {std:+.6f} ({rounds_b} rounds)")

    return mean, std


if __name__ == "__main__":
    honest = [0.10, 0.12, 0.09, 0.11, 0.10, 0.13]
    byzantine_signflip = [-1.5 * 0.115, -1.5 * 0.108]
    received = honest + byzantine_signflip

    print("=" * 70)
    print("PARALLEL TREE REDUCTION -- received values (honest + Byzantine)")
    print("=" * 70)
    mean_received, std_received = parallel_mean_and_std(received, trace=True)
    print(f"Final received mean={mean_received:.6f}, std={std_received:.6f}")

    print("=" * 70)
    print("PARALLEL TREE REDUCTION -- honest-only values")
    print("=" * 70)
    mean_honest, std_honest = parallel_mean_and_std(honest, trace=False)
    print(f"Final honest mean={mean_honest:.6f}, std={std_honest:.6f}")

    if torch.cuda.is_available() and ring_spmv_lib is not None:
        print("=" * 70)
        print("RING SpMV CONSENSUS -- one linear gossip round")
        print("=" * 70)

        device = "cuda"
        peer_vectors = [
            torch.tensor([x], device=device, dtype=torch.float32)
            for x in received
        ]

        engine = RingSpMVConsensus(num_streams=8, device=device)
        new_vectors = engine.step(peer_vectors)

        for i, nv in enumerate(new_vectors):
            print(f"Peer {i}: X_new = {nv.item():+.6f}")

        print("=" * 70)
        print("CONVERGENCE CHECK -- repeated linear gossip, no attack")
        print("=" * 70)

        clean_vectors = [
            torch.tensor([x], device=device, dtype=torch.float32)
            for x in honest + honest[:2]
        ]
        _, rounds_used = engine.run_until_converged(clean_vectors, eps=1e-2)
        print(f"Converged to within 1e-2 in {rounds_used} rounds")
    else:
        print("CUDA device or ring_spmv extension unavailable -- skipping GPU gossip demo.")
