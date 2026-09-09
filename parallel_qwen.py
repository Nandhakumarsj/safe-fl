#!/usr/bin/env python3
# generate_project.py
# Run this script to materialize the full project and create:
#   multi_stream_cuda_dfl_v2.zip

import os
import zipfile
from pathlib import Path

ROOT = Path("multi_stream_cuda_dfl_v2")
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "tests").mkdir(exist_ok=True)

FILES = {}

FILES["README.md"] = r'''Multi-Stream CUDA-Accelerated Decentralized Federated Learning Under Dual Threat Security Models

Secure next-version implementation.

Recommended secure mode:
    topology='full'
    update_type='gradient'

This mode supports coordinate-wise median robustness against up to 2 Byzantine peers out of 8.

Quick start:
    pip install torch numpy matplotlib
    python run_all_smoke.py
    python benchmark_aggregators.py
    python evaluate_and_plot.py
    python profile_runner.py
'''

FILES["requirements.txt"] = r'''torch>=2.0
numpy>=1.24
matplotlib>=3.7
'''

FILES["phase1_topology.py"] = r'''import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset


class SyntheticImageDataset(Dataset):
    def __init__(
        self,
        num_samples=256,
        num_classes=10,
        image_channels=3,
        image_size=16,
        sample_seed=0,
        template_seed=42,
    ):
        self.num_classes = num_classes

        template_generator = torch.Generator().manual_seed(template_seed)
        self.templates = torch.rand(
            num_classes,
            image_channels,
            image_size,
            image_size,
            generator=template_generator,
        )

        sample_generator = torch.Generator().manual_seed(sample_seed)
        self.targets = torch.randint(
            0,
            num_classes,
            (num_samples,),
            generator=sample_generator,
        )

        noise = torch.randn(
            num_samples,
            image_channels,
            image_size,
            image_size,
            generator=sample_generator,
        ) * 0.05

        self.data = torch.clamp(self.templates[self.targets] + noise, 0.0, 1.0)

    def __len__(self):
        return int(self.targets.numel())

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class RingTopology:
    def __init__(self, num_nodes):
        self.num_nodes = int(num_nodes)

    def get_neighbors(self, peer_id):
        peer_id = int(peer_id)
        left = (peer_id - 1) % self.num_nodes
        right = (peer_id + 1) % self.num_nodes
        return [left, peer_id, right]


class FullTopology:
    def __init__(self, num_nodes):
        self.num_nodes = int(num_nodes)

    def get_neighbors(self, peer_id):
        return list(range(self.num_nodes))


class LightweightCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Linear(32 * 2 * 2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_lightweight_cnn():
    return LightweightCNN(in_channels=3, num_classes=10)


class VirtualPeerManager:
    def __init__(
        self,
        model_fn=None,
        num_peers=8,
        device="cuda",
        lr=0.01,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.num_peers = int(num_peers)
        model_fn = model_fn or build_lightweight_cnn

        base_model = model_fn().to(self.device)

        self.models = []
        for _ in range(self.num_peers):
            model = model_fn().to(self.device)
            model.load_state_dict(base_model.state_dict())
            self.models.append(model)

        self.optimizers = [
            torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
            for model in self.models
        ]

        if self.device.startswith("cuda"):
            self.streams = [
                torch.cuda.Stream(device=self.device)
                for _ in range(self.num_peers)
            ]
        else:
            self.streams = [None for _ in range(self.num_peers)]

    def flatten_params(self, model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def unflatten_params(self, model, flat_params):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            slice_ = flat_params[offset:offset + numel].reshape_as(p)
            p.data.copy_(slice_.to(device=p.device, non_blocking=True))
            offset += numel

    def flatten_grads(self, model):
        flats = []
        for p in model.parameters():
            if p.grad is None:
                flats.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            else:
                flats.append(p.grad.detach().reshape(-1))
        return torch.cat(flats)

    def assign_flat_grads(self, model, flat_grads):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            grad_slice = flat_grads[offset:offset + numel].reshape_as(p)
            grad_slice = grad_slice.to(device=p.device, non_blocking=True)

            if p.grad is None:
                p.grad = grad_slice.clone()
            else:
                p.grad.copy_(grad_slice)

            offset += numel


class DataPartitioner:
    def __init__(
        self,
        num_peers=8,
        batch_size=64,
        image_channels=3,
        image_size=16,
        num_classes=10,
        train_samples_per_peer=256,
        test_samples=256,
        seed=0,
    ):
        self.num_peers = int(num_peers)
        self.batch_size = int(batch_size)
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.train_samples_per_peer = int(train_samples_per_peer)
        self.test_samples = int(test_samples)
        self.seed = int(seed)

    def _dataset(self, num_samples, sample_seed):
        return SyntheticImageDataset(
            num_samples=num_samples,
            num_classes=self.num_classes,
            image_channels=self.image_channels,
            image_size=self.image_size,
            sample_seed=sample_seed,
            template_seed=self.seed + 42,
        )

    def _subset_by_classes(self, dataset, classes):
        class_set = set(int(c) for c in classes)
        mask = torch.tensor([int(y.item() in class_set) for _, y in dataset], dtype=torch.bool)
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

        if len(indices) == 0:
            indices = list(range(len(dataset)))

        return Subset(dataset, indices)

    def get_peer_dataloaders(self, iid=True):
        test_dataset = self._dataset(
            num_samples=self.test_samples,
            sample_seed=self.seed + 999,
        )

        pin_memory = torch.cuda.is_available()

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        peer_loaders = []

        if iid:
            for peer_id in range(self.num_peers):
                dataset = self._dataset(
                    num_samples=self.train_samples_per_peer,
                    sample_seed=self.seed + 1000 + peer_id,
                )
                peer_loaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=pin_memory,
                    )
                )
        else:
            global_dataset = self._dataset(
                num_samples=self.train_samples_per_peer * self.num_peers,
                sample_seed=self.seed + 5000,
            )

            for peer_id in range(self.num_peers):
                classes = [
                    peer_id % self.num_classes,
                    (peer_id + 1) % self.num_classes,
                ]
                subset = self._subset_by_classes(global_dataset, classes)
                peer_loaders.append(
                    DataLoader(
                        subset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=pin_memory,
                    )
                )

        return peer_loaders, test_loader
'''

FILES["phase2_security.py"] = r'''import contextlib
import torch
import torch.nn as nn


class PGD10Attacker:
    def __init__(self, eps=8 / 255, alpha=2 / 255, steps=10):
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, model, x, y, stream=None):
        if stream is not None and torch.cuda.is_available():
            ctx = torch.cuda.stream(stream)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            was_training = model.training
            model.eval()

            x_orig = x.detach().clone()
            delta = torch.empty_like(x_orig).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

            for _ in range(self.steps):
                x_adv.requires_grad_(True)

                outputs = model(x_adv)
                loss = self.loss_fn(outputs, y)

                grad = torch.autograd.grad(
                    outputs=loss,
                    inputs=x_adv,
                    only_inputs=True,
                )[0]

                x_adv = x_adv.detach() + self.alpha * grad.sign()
                eta = torch.clamp(x_adv - x_orig, min=-self.eps, max=self.eps)
                x_adv = torch.clamp(x_orig + eta, min=0.0, max=1.0).detach()

            if was_training:
                model.train()

            return x_adv


class ByzantineAttackEngine:
    def __init__(self, num_peers=8, byzantine_indices=(6, 7), attack_type="sign_flip"):
        self.num_peers = int(num_peers)
        self.byzantine_indices = set(int(i) for i in byzantine_indices)
        self.attack_type = attack_type

    def _honest_stats(self, updates):
        honest_idx = [
            i for i in range(self.num_peers)
            if i not in self.byzantine_indices
        ]

        if len(honest_idx) == 0:
            raise ValueError("No honest peers available for Byzantine statistics.")

        honest_stack = torch.stack([updates[i] for i in honest_idx], dim=0)
        mean_honest = torch.mean(honest_stack, dim=0)
        std_honest = torch.std(honest_stack, dim=0, unbiased=False) + 1e-8

        return mean_honest, std_honest

    def poison(self, honest_updates, attack_type=None):
        attack = attack_type or self.attack_type

        if len(honest_updates) != self.num_peers:
            raise ValueError(
                f"Expected {self.num_peers} updates, got {len(honest_updates)}"
            )

        if not self.byzantine_indices:
            return [u.clone() for u in honest_updates]

        mean_honest, std_honest = self._honest_stats(honest_updates)
        poisoned = []

        for i in range(self.num_peers):
            update = honest_updates[i]

            if i not in self.byzantine_indices:
                poisoned.append(update.clone())
                continue

            if attack == "sign_flip":
                poisoned.append(-1.5 * update)

            elif attack == "gaussian_noise":
                poisoned.append(
                    mean_honest + torch.randn_like(update) * 3.0 * std_honest
                )

            elif attack == "zero_gradient":
                poisoned.append(torch.zeros_like(update))

            elif attack == "alie":
                z_max = 0.84
                poisoned.append(mean_honest - z_max * std_honest)

            elif attack == "targeted_dimension":
                poisoned_vector = update.clone()
                top_k = max(1, int(0.05 * update.numel()))
                top_k = min(top_k, update.numel())

                _, top_indices = torch.topk(std_honest, top_k)
                poisoned_vector[top_indices] = -3.0 * mean_honest[top_indices]
                poisoned.append(poisoned_vector)

            else:
                raise ValueError(f"Unsupported attack_type: {attack}")

        return poisoned
'''

FILES["aggregators.py"] = r'''import torch


class AggregatorSuite:
    @staticmethod
    def median(neighbor_updates):
        stacked = torch.stack(neighbor_updates, dim=0)
        return torch.median(stacked, dim=0).values

    @staticmethod
    def fed_avg(neighbor_updates):
        stacked = torch.stack(neighbor_updates, dim=0)
        return torch.mean(stacked, dim=0)

    @staticmethod
    def trimmed_mean(neighbor_updates, trim_count=None):
        stacked = torch.stack(neighbor_updates, dim=0)
        K = stacked.size(0)

        if trim_count is None:
            trim_count = 2 if K >= 8 else 1

        trim_count = min(int(trim_count), (K - 1) // 2)

        sorted_t, _ = torch.sort(stacked, dim=0)

        if trim_count == 0:
            return torch.mean(sorted_t, dim=0)

        return torch.mean(sorted_t[trim_count: K - trim_count, :], dim=0)

    @staticmethod
    def krum(neighbor_updates, num_byz=None):
        stacked = torch.stack(neighbor_updates, dim=0)
        K = stacked.size(0)

        if num_byz is None:
            num_byz = 2 if K >= 8 else 1

        keep = max(1, K - int(num_byz) - 2)

        distances = torch.cdist(stacked, stacked, p=2).pow(2)

        scores = []
        for i in range(K):
            sorted_d, _ = torch.sort(distances[i])
            scores.append(torch.sum(sorted_d[1:1 + keep]))

        best_idx = int(torch.argmin(torch.stack(scores)))
        return stacked[best_idx]
'''

FILES["phase3_consensus.py"] = r'''import os
import warnings

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_extension():
    if not torch.cuda.is_available():
        return None

    try:
        return load(
            name="consensus_cuda",
            sources=[
                os.path.join(_THIS_DIR, "consensus_kernel.cu"),
                os.path.join(_THIS_DIR, "consensus_binding.cpp"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
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
'''

FILES["consensus_kernel.cu"] = r'''#include <cuda_runtime.h>

__device__ __forceinline__ float median3(float a, float b, float c) {
    float mn = fminf(a, fminf(b, c));
    float mx = fmaxf(a, fmaxf(b, c));
    return a + b + c - mn - mx;
}

__device__ __forceinline__ void sort8(
    float &v0, float &v1, float &v2, float &v3,
    float &v4, float &v5, float &v6, float &v7
) {
#define MINMAX(a, b) \
    { \
        float t = a; \
        a = fminf(a, b); \
        b = fmaxf(t, b); \
    }

    MINMAX(v0, v1); MINMAX(v2, v3); MINMAX(v4, v5); MINMAX(v6, v7);
    MINMAX(v0, v2); MINMAX(v1, v3); MINMAX(v4, v6); MINMAX(v5, v7);
    MINMAX(v1, v2); MINMAX(v5, v6); MINMAX(v0, v4); MINMAX(v3, v7);
    MINMAX(v1, v5); MINMAX(v2, v6);
    MINMAX(v1, v4); MINMAX(v3, v6);
    MINMAX(v2, v4); MINMAX(v3, v5);
    MINMAX(v3, v4);

#undef MINMAX
}

__global__ void median3_kernel(
    const float* __restrict__ gradients,
    float* __restrict__ aggregated,
    int start,
    int end
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end) {
        const float* row = gradients + idx * 3;
        aggregated[idx] = median3(row[0], row[1], row[2]);
    }
}

__global__ void median8_kernel(
    const float* __restrict__ gradients,
    float* __restrict__ aggregated,
    int start,
    int end
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end) {
        const float* row = gradients + idx * 8;

        float v0 = row[0];
        float v1 = row[1];
        float v2 = row[2];
        float v3 = row[3];
        float v4 = row[4];
        float v5 = row[5];
        float v6 = row[6];
        float v7 = row[7];

        sort8(v0, v1, v2, v3, v4, v5, v6, v7);

        aggregated[idx] = 0.5f * (v3 + v4);
    }
}

extern "C" void launch_coordinate_wise_median(
    const float* gradients_ptr,
    float* aggregated_ptr,
    int num_coordinates,
    int num_clients,
    int num_streams,
    const cudaStream_t* streams
) {
    const int threads = 256;
    int total = num_coordinates;

    int chunk = (total + num_streams - 1) / num_streams;

    for (int sid = 0; sid < num_streams; ++sid) {
        int start = sid * chunk;
        if (start >= total) {
            break;
        }

        int remaining = total - start;
        int count = (chunk < remaining) ? chunk : remaining;
        int end = start + count;
        int blocks = (count + threads - 1) / threads;

        if (num_clients == 3) {
            median3_kernel<<<blocks, threads, 0, streams[sid]>>>(
                gradients_ptr,
                aggregated_ptr,
                start,
                end
            );
        } else if (num_clients == 8) {
            median8_kernel<<<blocks, threads, 0, streams[sid]>>>(
                gradients_ptr,
                aggregated_ptr,
                start,
                end
            );
        }
    }
}
'''

FILES["consensus_binding.cpp"] = r'''#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void launch_coordinate_wise_median(
    const float* gradients_ptr,
    float* aggregated_ptr,
    int num_coordinates,
    int num_clients,
    int num_streams,
    const cudaStream_t* streams
);

torch::Tensor coordinate_wise_median(
    torch::Tensor gradients,
    std::vector<int64_t> stream_ptrs
) {
    TORCH_CHECK(gradients.dim() == 2, "gradients must be a 2D tensor [D, K]");
    TORCH_CHECK(gradients.dtype() == torch::kFloat32, "gradients must be float32");
    TORCH_CHECK(gradients.is_cuda(), "gradients must be on a CUDA device");
    TORCH_CHECK(
        gradients.is_contiguous(),
        "gradients must be contiguous (call .contiguous() after transpose)"
    );

    int64_t D = gradients.size(0);
    int64_t K = gradients.size(1);

    TORCH_CHECK(
        K == 3 || K == 8,
        "num_clients (K) must be 3 (ring neighbourhood) or 8 (global), got ",
        K
    );

    int num_streams = static_cast<int>(stream_ptrs.size());
    TORCH_CHECK(num_streams > 0, "at least one CUDA stream handle must be supplied");

    at::cuda::CUDAGuard device_guard(gradients.device().index());

    torch::Tensor aggregated = torch::empty({D}, gradients.options());

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        streams[i] = reinterpret_cast<cudaStream_t>(stream_ptrs[i]);
    }

    launch_coordinate_wise_median(
        gradients.data_ptr<float>(),
        aggregated.data_ptr<float>(),
        static_cast<int>(D),
        static_cast<int>(K),
        num_streams,
        streams.data()
    );

    C10_CUDA_CHECK(cudaGetLastError());

    return aggregated;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "coordinate_wise_median",
        &coordinate_wise_median,
        "Coordinate-wise median aggregation over K=3 or K=8 neighbours (CUDA)",
        py::arg("gradients"),
        py::arg("stream_ptrs")
    );
}
'''

FILES["phase3b_linear_consensus.py"] = r'''import os
import math
import warnings

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_ring_spmv():
    if not torch.cuda.is_available():
        return None

    try:
        return load(
            name="ring_spmv_cuda",
            sources=[
                os.path.join(_THIS_DIR, "ring_spmv_kernel.cu"),
                os.path.join(_THIS_DIR, "ring_spmv_binding.cpp"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
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
'''

FILES["ring_spmv_kernel.cu"] = r'''#include <cuda_runtime.h>

__global__ void ring_spmv_kernel(
    const float* __restrict__ X,
    float* __restrict__ X_new,
    int N,
    int D,
    int start,
    int end
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end) {
        int row = idx / D;
        int col = idx - row * D;

        int left = (row == 0) ? (N - 1) : (row - 1);
        int right = (row == N - 1) ? 0 : (row + 1);

        float self_val = X[idx];
        float left_val = X[left * D + col];
        float right_val = X[right * D + col];

        X_new[idx] = (left_val + self_val + right_val) / 3.0f;
    }
}

extern "C" void launch_ring_spmv(
    const float* X_ptr,
    float* X_new_ptr,
    int N,
    int D,
    int num_streams,
    const cudaStream_t* streams
) {
    const int threads = 256;
    int total = N * D;

    int chunk = (total + num_streams - 1) / num_streams;

    for (int sid = 0; sid < num_streams; ++sid) {
        int start = sid * chunk;
        if (start >= total) {
            break;
        }

        int remaining = total - start;
        int count = (chunk < remaining) ? chunk : remaining;
        int end = start + count;
        int blocks = (count + threads - 1) / threads;

        ring_spmv_kernel<<<blocks, threads, 0, streams[sid]>>>(
            X_ptr,
            X_new_ptr,
            N,
            D,
            start,
            end
        );
    }
}
'''

FILES["ring_spmv_binding.cpp"] = r'''#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void launch_ring_spmv(
    const float* X_ptr,
    float* X_new_ptr,
    int N,
    int D,
    int num_streams,
    const cudaStream_t* streams
);

torch::Tensor ring_spmv(
    torch::Tensor X,
    std::vector<int64_t> stream_ptrs
) {
    TORCH_CHECK(X.dim() == 2, "X must be a 2D tensor [N, D]");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.is_cuda(), "X must be on a CUDA device");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    int64_t N = X.size(0);
    int64_t D = X.size(1);

    TORCH_CHECK(N >= 3, "ring topology requires at least 3 peers, got ", N);

    int num_streams = static_cast<int>(stream_ptrs.size());
    TORCH_CHECK(num_streams > 0, "at least one CUDA stream handle must be supplied");

    at::cuda::CUDAGuard device_guard(X.device().index());

    torch::Tensor X_new = torch::empty_like(X);

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        streams[i] = reinterpret_cast<cudaStream_t>(stream_ptrs[i]);
    }

    launch_ring_spmv(
        X.data_ptr<float>(),
        X_new.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(D),
        num_streams,
        streams.data()
    );

    C10_CUDA_CHECK(cudaGetLastError());

    return X_new;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ring_spmv",
        &ring_spmv,
        "One round of ring-topology linear gossip consensus X_new = W.X (CUDA)",
        py::arg("X"),
        py::arg("stream_ptrs")
    );
}
'''

FILES["end_to_end_fl.py"] = r'''import contextlib
import time

import torch
import torch.nn as nn

from phase1_topology import (
    RingTopology,
    FullTopology,
    VirtualPeerManager,
    DataPartitioner,
    build_lightweight_cnn,
)
from phase2_security import PGD10Attacker, ByzantineAttackEngine
from phase3_consensus import MultiStreamConsensus
from aggregators import AggregatorSuite


def _nvtx_push(name, enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop(enabled):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


class DecentralizedTrainer:
    def __init__(
        self,
        num_peers=8,
        byzantine_indices=(6, 7),
        attack_type="sign_flip",
        model_fn=None,
        device="cuda",
        topology="full",
        update_type="gradient",
        use_pgd=True,
        lr=0.01,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.num_peers = int(num_peers)
        self.device = device
        self.update_type = update_type
        self.use_pgd = bool(use_pgd)

        if update_type not in {"gradient", "delta", "weight"}:
            raise ValueError("update_type must be one of: gradient, delta, weight")

        model_fn = model_fn or build_lightweight_cnn

        print("[+] Phase 1: Virtual Peer Harness & Topology...")
        self.vpm = VirtualPeerManager(
            model_fn=model_fn,
            num_peers=num_peers,
            device=device,
            lr=lr,
        )

        if topology == "full":
            self.topology = FullTopology(num_nodes=num_peers)
            print("[+] Topology: FULL, K=8 global consensus")
        elif topology == "ring":
            self.topology = RingTopology(num_nodes=num_peers)
            print("[+] Topology: RING, K=3 local consensus")
        else:
            raise ValueError("topology must be 'full' or 'ring'")

        print("[+] Phase 2: PGD-10 Attacker & Byzantine Engine...")
        self.pgd_attacker = PGD10Attacker()
        self.byzantine_engine = ByzantineAttackEngine(
            num_peers=num_peers,
            byzantine_indices=byzantine_indices,
            attack_type=attack_type,
        )

        print("[+] Phase 3: CUDA Median Consensus Engine...")
        self.consensus_engine = MultiStreamConsensus(
            num_streams=num_peers,
            device=device,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def _stream_context(self, stream):
        if stream is None:
            return contextlib.nullcontext()
        return torch.cuda.stream(stream)

    def _aggregate_single_peer(self, neighbor_updates, aggregator):
        if aggregator == "cuda_median":
            return self.consensus_engine.aggregate_neighbors(neighbor_updates)
        if aggregator == "median":
            return AggregatorSuite.median(neighbor_updates)
        if aggregator == "trimmed_mean":
            return AggregatorSuite.trimmed_mean(neighbor_updates)
        if aggregator == "krum":
            return AggregatorSuite.krum(neighbor_updates)
        if aggregator == "fed_avg":
            return AggregatorSuite.fed_avg(neighbor_updates)

        raise ValueError(f"Unsupported aggregator: {aggregator}")

    def train_one_batch(
        self,
        loaders_iter,
        aggregator="cuda_median",
        use_byzantine=True,
        enable_nvtx=False,
    ):
        honest_updates = []
        old_params = []

        _nvtx_push("Phase2_PGD10_Local_Training", enable_nvtx)

        for peer_id in range(self.num_peers):
            _nvtx_push(f"Peer_{peer_id}_Local", enable_nvtx)

            stream = self.vpm.streams[peer_id]
            model = self.vpm.models[peer_id]
            optimizer = self.vpm.optimizers[peer_id]

            x, y = next(loaders_iter[peer_id])

            with self._stream_context(stream):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.use_pgd:
                    x_adv = self.pgd_attacker.perturb(
                        model,
                        x,
                        y,
                        stream=stream,
                    )
                else:
                    x_adv = x

                optimizer.zero_grad(set_to_none=True)

                outputs = model(x_adv)
                loss = self.loss_fn(outputs, y)
                loss.backward()

                if self.update_type == "gradient":
                    honest_updates.append(self.vpm.flatten_grads(model))

                elif self.update_type == "delta":
                    old_flat = self.vpm.flatten_params(model)
                    old_params.append(old_flat)

                    optimizer.step()

                    new_flat = self.vpm.flatten_params(model)
                    honest_updates.append(new_flat - old_flat)

                else:
                    optimizer.step()
                    honest_updates.append(self.vpm.flatten_params(model))

            _nvtx_pop(enable_nvtx)

        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

        _nvtx_pop(enable_nvtx)

        _nvtx_push("Phase2_Byzantine_Poisoning", enable_nvtx)

        if use_byzantine:
            poisoned_updates = self.byzantine_engine.poison(honest_updates)
        else:
            poisoned_updates = [u.clone() for u in honest_updates]

        _nvtx_pop(enable_nvtx)

        _nvtx_push("Phase3_Consensus_Aggregation", enable_nvtx)

        robust_updates = []

        for peer_id in range(self.num_peers):
            _nvtx_push(f"Peer_{peer_id}_Gossip_Aggregation", enable_nvtx)

            neighbors = self.topology.get_neighbors(peer_id)
            neighbor_updates = [poisoned_updates[n] for n in neighbors]

            robust = self._aggregate_single_peer(neighbor_updates, aggregator)
            robust_updates.append(robust)

            _nvtx_pop(enable_nvtx)

        _nvtx_pop(enable_nvtx)

        _nvtx_push("Phase4_Apply_Robust_Updates", enable_nvtx)

        if self.update_type == "gradient":
            for peer_id in range(self.num_peers):
                self.vpm.assign_flat_grads(
                    self.vpm.models[peer_id],
                    robust_updates[peer_id],
                )
                self.vpm.optimizers[peer_id].step()

        elif self.update_type == "delta":
            for peer_id in range(self.num_peers):
                new_flat = old_params[peer_id] + robust_updates[peer_id]
                self.vpm.unflatten_params(
                    self.vpm.models[peer_id],
                    new_flat,
                )

        else:
            for peer_id in range(self.num_peers):
                self.vpm.unflatten_params(
                    self.vpm.models[peer_id],
                    robust_updates[peer_id],
                )

        _nvtx_pop(enable_nvtx)

    def train_epoch(
        self,
        peer_dataloaders,
        aggregator="cuda_median",
        use_byzantine=True,
        enable_nvtx=False,
    ):
        epoch_start = time.time()

        min_batches = min(len(dl) for dl in peer_dataloaders)
        loaders_iter = [iter(dl) for dl in peer_dataloaders]

        for _ in range(min_batches):
            self.train_one_batch(
                loaders_iter,
                aggregator=aggregator,
                use_byzantine=use_byzantine,
                enable_nvtx=enable_nvtx,
            )

        return time.time() - epoch_start


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cuda":
        print("CUDA device not available -- running CPU fallback smoke test.")

    num_peers = 8

    partitioner = DataPartitioner(
        num_peers=num_peers,
        batch_size=32,
        train_samples_per_peer=128,
        test_samples=128,
    )

    peer_loaders, _ = partitioner.get_peer_dataloaders(iid=True)

    orchestrator = DecentralizedTrainer(
        num_peers=num_peers,
        byzantine_indices=(6, 7),
        attack_type="sign_flip",
        device=device,
        topology="full",
        update_type="gradient",
        use_pgd=True,
    )

    print("\n[+] Starting End-to-End FL Loop Validation (1 epoch)...")
    latency = orchestrator.train_epoch(peer_loaders, aggregator="cuda_median")

    print(
        f"\n[OK] Training epoch completed in {latency:.2f}s "
        f"({num_peers} peers, 2 Byzantine nodes defended, PGD-10 + CUDA-median active)."
    )
'''

FILES["benchmark_aggregators.py"] = r'''import numpy as np
import torch

from end_to_end_fl import DecentralizedTrainer
from phase1_topology import DataPartitioner
from phase2_security import PGD10Attacker


def evaluate_model(model, test_loader, attacker=None, device="cuda"):
    model.eval()
    correct, total = 0, 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_eval = attacker.perturb(model, x, y) if attacker is not None else x

        with torch.no_grad():
            preds = model(x_eval).argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    model.train()
    return 100.0 * correct / total


def run_benchmark_matrix(
    epochs=1,
    device="cuda",
    num_peers=8,
    byzantine_indices=(6, 7),
    topology="full",
    update_type="gradient",
):
    partitioner = DataPartitioner(
        num_peers=num_peers,
        batch_size=64,
        train_samples_per_peer=256,
        test_samples=256,
    )
    peer_loaders, test_loader = partitioner.get_peer_dataloaders(iid=True)

    attacks = ["sign_flip", "alie", "targeted_dimension"]
    aggregators = ["cuda_median", "trimmed_mean", "krum", "fed_avg"]

    eval_attacker = PGD10Attacker(eps=8 / 255, alpha=2 / 255, steps=10)
    honest_indices = [i for i in range(num_peers) if i not in set(byzantine_indices)]

    results = {}

    for attack in attacks:
        results[attack] = {}

        for agg in aggregators:
            print(f"\n[+] Defense [{agg.upper()}] vs. Attack [{attack.upper()}]")

            trainer = DecentralizedTrainer(
                num_peers=num_peers,
                byzantine_indices=byzantine_indices,
                attack_type=attack,
                device=device,
                topology=topology,
                update_type=update_type,
            )

            for _ in range(int(epochs)):
                trainer.train_epoch(
                    peer_loaders,
                    aggregator=agg,
                    use_byzantine=True,
                )

            clean_accs = [
                evaluate_model(trainer.vpm.models[p], test_loader, None, device)
                for p in honest_indices
            ]
            adv_accs = [
                evaluate_model(trainer.vpm.models[p], test_loader, eval_attacker, device)
                for p in honest_indices
            ]

            results[attack][agg] = {
                "clean_acc": float(np.mean(clean_accs)),
                "adv_acc": float(np.mean(adv_accs)),
            }

    return results


def print_summary_table(results):
    print()
    print("=" * 78)
    print("BENCHMARK RESULTS SUMMARY: DEFENSES VS. BYZANTINE ATTACKS")
    print("=" * 78)
    print(f"{'Attack Type':<20} | {'Aggregator':<15} | {'Clean Acc (%)':<16} | {'PGD-10 Acc (%)':<16}")
    print("-" * 78)

    for attack, aggs in results.items():
        for agg_name, metrics in aggs.items():
            print(
                f"{attack:<20} | {agg_name:<15} | "
                f"{metrics['clean_acc']:<16.2f} | {metrics['adv_acc']:<16.2f}"
            )

    print("-" * 78)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    benchmark_results = run_benchmark_matrix(
        epochs=1,
        device=device,
        topology="full",
        update_type="gradient",
    )

    print_summary_table(benchmark_results)
'''

FILES["evaluate_and_plot.py"] = r'''import time
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from end_to_end_fl import DecentralizedTrainer
from phase1_topology import DataPartitioner
from phase2_security import PGD10Attacker


def evaluate(model, test_loader, attacker=None, device="cuda"):
    model.eval()
    correct, total = 0, 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_eval = attacker.perturb(model, x, y) if attacker is not None else x

        with torch.no_grad():
            preds = model(x_eval).argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    model.train()
    return 100.0 * correct / total


def run_evaluation(
    device="cuda",
    num_peers=8,
    byzantine_indices=(6, 7),
    epochs=3,
    topology="full",
    update_type="gradient",
):
    partitioner = DataPartitioner(
        num_peers=num_peers,
        batch_size=64,
        train_samples_per_peer=256,
        test_samples=256,
    )

    peer_loaders, test_loader = partitioner.get_peer_dataloaders(iid=True)

    trainer = DecentralizedTrainer(
        num_peers=num_peers,
        byzantine_indices=byzantine_indices,
        attack_type="sign_flip",
        device=device,
        topology=topology,
        update_type=update_type,
    )

    eval_attacker = PGD10Attacker(eps=8 / 255, alpha=2 / 255, steps=10)
    honest_indices = [i for i in range(num_peers) if i not in set(byzantine_indices)]

    clean_history, adv_history = [], []

    print(f"\nStarting {epochs}-epoch evaluation (sign-flip attack on peers {list(byzantine_indices)})\n")

    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()

        trainer.train_epoch(
            peer_loaders,
            aggregator="cuda_median",
            use_byzantine=True,
        )

        elapsed = time.time() - t0

        clean_accs = [
            evaluate(trainer.vpm.models[p], test_loader, None, device)
            for p in honest_indices
        ]
        adv_accs = [
            evaluate(trainer.vpm.models[p], test_loader, eval_attacker, device)
            for p in honest_indices
        ]

        avg_clean = float(np.mean(clean_accs))
        avg_adv = float(np.mean(adv_accs))

        clean_history.append(avg_clean)
        adv_history.append(avg_adv)

        print(
            f"Epoch {epoch}/{epochs} [{elapsed:.1f}s] | "
            f"Clean Acc: {avg_clean:.2f}% | PGD-10 Acc: {avg_adv:.2f}%"
        )

    return clean_history, adv_history


def plot_publication_figure(
    epochs,
    clean_accs,
    adv_accs,
    out_prefix="fl_byzantine_robustness",
):
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)

    ax.plot(
        epochs,
        clean_accs,
        marker="o",
        label="Clean Accuracy (Honest Nodes)",
        color="#1f77b4",
    )

    ax.plot(
        epochs,
        adv_accs,
        marker="s",
        label=r"PGD-10 Adversarial Accuracy ($\epsilon=8/255$)",
        color="#d62728",
    )

    ax.set_xlabel("Training Epochs", fontweight="bold")
    ax.set_ylabel("Top-1 Test Accuracy (%)", fontweight="bold")
    ax.set_title(
        "Decentralized FL Robustness under Sign-Flip Byzantine Attack\n"
        "(N=8 Peers, 2 Byzantine, CUDA Median Consensus)"
    )

    ax.set_xticks(epochs)
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved {out_prefix}.png / {out_prefix}.pdf")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs_list = list(range(1, 4))
    clean_accs, adv_accs = run_evaluation(device=device, epochs=3)

    plot_publication_figure(epochs_list, clean_accs, adv_accs)
'''

FILES["profile_runner.py"] = r'''import torch

from end_to_end_fl import DecentralizedTrainer
from phase1_topology import DataPartitioner


def run_profiled_iterations(
    trainer,
    peer_loaders,
    warmup_steps=2,
    profile_steps=5,
    aggregator="cuda_median",
):
    def make_iters():
        return [iter(dl) for dl in peer_loaders]

    loaders_iter = make_iters()

    def run_step(enable_nvtx):
        nonlocal loaders_iter
        try:
            trainer.train_one_batch(
                loaders_iter,
                aggregator=aggregator,
                use_byzantine=True,
                enable_nvtx=enable_nvtx,
            )
        except StopIteration:
            loaders_iter = make_iters()
            trainer.train_one_batch(
                loaders_iter,
                aggregator=aggregator,
                use_byzantine=True,
                enable_nvtx=enable_nvtx,
            )

    print(f"[+] Running {warmup_steps} warm-up steps...")
    for _ in range(int(warmup_steps)):
        run_step(enable_nvtx=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"[+] Starting {profile_steps} NVTX-instrumented profiling steps...")

    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()

    for step in range(int(profile_steps)):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(f"Profiled_Batch_Step_{step}")

        run_step(enable_nvtx=True)

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

    if torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()

    print("[OK] Profiling run completed successfully.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_peers = 8

    partitioner = DataPartitioner(
        num_peers=num_peers,
        batch_size=32,
        train_samples_per_peer=256,
        test_samples=128,
    )

    peer_loaders, _ = partitioner.get_peer_dataloaders(iid=True)

    trainer = DecentralizedTrainer(
        num_peers=num_peers,
        byzantine_indices=(6, 7),
        attack_type="sign_flip",
        device=device,
        topology="full",
        update_type="gradient",
        use_pgd=True,
    )

    run_profiled_iterations(
        trainer,
        peer_loaders,
        warmup_steps=1,
        profile_steps=2,
        aggregator="cuda_median",
    )
'''

FILES["run_all_smoke.py"] = r'''import torch

from phase1_topology import DataPartitioner
from end_to_end_fl import DecentralizedTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[smoke] device = {device}")

    partitioner = DataPartitioner(
        num_peers=8,
        batch_size=16,
        train_samples_per_peer=64,
        test_samples=32,
        image_size=8,
    )

    peer_loaders, test_loader = partitioner.get_peer_dataloaders(iid=True)

    trainer = DecentralizedTrainer(
        num_peers=8,
        byzantine_indices=(6, 7),
        attack_type="sign_flip",
        device=device,
        topology="full",
        update_type="gradient",
        use_pgd=True,
    )

    elapsed = trainer.train_epoch(
        peer_loaders,
        aggregator="cuda_median",
        use_byzantine=True,
    )

    print(f"[smoke] one epoch completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
'''

FILES["tests/__init__.py"] = ""

FILES["tests/test_core.py"] = r'''import unittest

import torch

from phase1_topology import (
    RingTopology,
    FullTopology,
    VirtualPeerManager,
    DataPartitioner,
    build_lightweight_cnn,
)
from phase2_security import ByzantineAttackEngine
from aggregators import AggregatorSuite


class TestTopologies(unittest.TestCase):
    def test_ring_neighbors(self):
        topo = RingTopology(num_nodes=8)
        neighbors = topo.get_neighbors(0)
        self.assertEqual(len(neighbors), 3)
        self.assertIn(0, neighbors)
        self.assertIn(7, neighbors)
        self.assertIn(1, neighbors)

    def test_full_neighbors(self):
        topo = FullTopology(num_nodes=8)
        neighbors = topo.get_neighbors(3)
        self.assertEqual(len(neighbors), 8)
        self.assertEqual(neighbors, list(range(8)))


class TestAggregators(unittest.TestCase):
    def test_median_blocks_two_byzantine_out_of_eight(self):
        D = 100
        honest = [torch.ones(D) for _ in range(6)]
        malicious = [-100.0 * torch.ones(D) for _ in range(2)]
        updates = honest + malicious

        median = AggregatorSuite.median(updates)
        self.assertTrue(torch.allclose(median, torch.ones(D)))

    def test_trimmed_mean_k8(self):
        D = 50
        updates = [torch.ones(D) + 0.01 * i for i in range(6)]
        updates += [-100.0 * torch.ones(D) for _ in range(2)]

        trimmed = AggregatorSuite.trimmed_mean(updates)
        self.assertTrue(trimmed.mean().item() > 0.0)
        self.assertTrue(trimmed.mean().item() < 10.0)

    def test_krum_returns_valid_vector(self):
        D = 20
        updates = [torch.ones(D) + 0.001 * i for i in range(8)]
        updates[7] = -50.0 * torch.ones(D)

        chosen = AggregatorSuite.krum(updates)
        self.assertEqual(chosen.shape[0], D)


class TestByzantineEngine(unittest.TestCase):
    def test_poison_length_and_byzantine_modification(self):
        engine = ByzantineAttackEngine(
            num_peers=8,
            byzantine_indices=(6, 7),
            attack_type="sign_flip",
        )

        updates = [torch.ones(64) + 0.01 * i for i in range(8)]
        poisoned = engine.poison(updates)

        self.assertEqual(len(poisoned), 8)
        self.assertTrue(torch.allclose(poisoned[0], updates[0]))
        self.assertTrue((poisoned[6] < 0).any())

    def test_all_supported_attacks_run(self):
        attacks = [
            "sign_flip",
            "gaussian_noise",
            "zero_gradient",
            "alie",
            "targeted_dimension",
        ]

        updates = [torch.randn(32) for _ in range(8)]

        for attack in attacks:
            engine = ByzantineAttackEngine(
                num_peers=8,
                byzantine_indices=(6, 7),
                attack_type=attack,
            )
            poisoned = engine.poison(updates)
            self.assertEqual(len(poisoned), 8)


class TestVirtualPeerManager(unittest.TestCase):
    def test_flatten_unflatten_roundtrip_cpu(self):
        vpm = VirtualPeerManager(
            model_fn=build_lightweight_cnn,
            num_peers=2,
            device="cpu",
        )

        model = vpm.models[0]
        flat = vpm.flatten_params(model)
        flat_copy = flat.clone()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        vpm.unflatten_params(model, flat_copy)

        flat_after = vpm.flatten_params(model)
        self.assertTrue(torch.allclose(flat_after, flat_copy))

    def test_data_partitioner_cpu(self):
        partitioner = DataPartitioner(
            num_peers=4,
            batch_size=8,
            train_samples_per_peer=32,
            test_samples=16,
            image_size=8,
        )

        peer_loaders, test_loader = partitioner.get_peer_dataloaders(iid=True)
        self.assertEqual(len(peer_loaders), 4)

        x, y = next(iter(peer_loaders[0]))
        self.assertEqual(x.shape[1], 3)
        self.assertEqual(x.shape[2], 8)
        self.assertEqual(x.shape[3], 8)


if __name__ == "__main__":
    unittest.main()
'''

for rel_path, content in FILES.items():
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

zip_path = Path("multi_stream_cuda_dfl_v2.zip")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for file_path in sorted(ROOT.rglob("*")):
        if file_path.is_file():
            zf.write(file_path, file_path.relative_to(ROOT.parent))

print(f"Created {zip_path.resolve()}")