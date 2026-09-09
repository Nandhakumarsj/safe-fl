import contextlib
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
