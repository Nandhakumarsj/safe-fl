import unittest

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
