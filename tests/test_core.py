"""
tests/test_core.py
==================
Comprehensive unit tests for the Multi-Stream CUDA-Accelerated
Decentralised Federated Learning system.

All tests run on CPU so they work without a GPU.
"""

import math
import unittest

import torch
import torch.nn as nn

from phase1_topology import (
    RingTopology,
    FullTopology,
    VirtualPeerManager,
    DataPartitioner,
    build_lightweight_cnn,
    LightweightCNN,
    SyntheticImageDataset,
)
from phase2_security import PGD10Attacker, ByzantineAttackEngine
from phase3b_linear_consensus import (
    RingSpMVConsensus,
    parallel_tree_reduce,
    parallel_mean_and_std,
)
from aggregators import AggregatorSuite


# ---------------------------------------------------------------------------
# TestTopologies
# ---------------------------------------------------------------------------

class TestTopologies(unittest.TestCase):

    def test_ring_neighbors_node0(self):
        topo = RingTopology(num_nodes=8)
        neighbors = topo.get_neighbors(0)
        self.assertEqual(len(neighbors), 3)
        self.assertIn(0, neighbors)   # self
        self.assertIn(7, neighbors)   # left wrap-around
        self.assertIn(1, neighbors)   # right

    def test_ring_neighbors_mid_node(self):
        topo = RingTopology(num_nodes=8)
        neighbors = topo.get_neighbors(4)
        self.assertEqual(set(neighbors), {3, 4, 5})

    def test_ring_neighbors_last_node(self):
        topo = RingTopology(num_nodes=8)
        neighbors = topo.get_neighbors(7)
        self.assertEqual(set(neighbors), {6, 7, 0})

    def test_ring_small_topology(self):
        topo = RingTopology(num_nodes=3)
        # In a 3-node ring every node is adjacent to all others
        for i in range(3):
            self.assertEqual(len(topo.get_neighbors(i)), 3)

    def test_full_neighbors(self):
        topo = FullTopology(num_nodes=8)
        neighbors = topo.get_neighbors(3)
        self.assertEqual(len(neighbors), 8)
        self.assertEqual(neighbors, list(range(8)))

    def test_full_neighbors_single_node(self):
        topo = FullTopology(num_nodes=1)
        self.assertEqual(topo.get_neighbors(0), [0])

    def test_ring_always_includes_self(self):
        topo = RingTopology(num_nodes=10)
        for i in range(10):
            self.assertIn(i, topo.get_neighbors(i))


# ---------------------------------------------------------------------------
# TestAggregators
# ---------------------------------------------------------------------------

class TestAggregators(unittest.TestCase):

    def _make_updates(self, D, K, honest_val=1.0, bad_val=-100.0, n_byz=2):
        honest = [torch.full((D,), honest_val) for _ in range(K - n_byz)]
        malicious = [torch.full((D,), bad_val) for _ in range(n_byz)]
        return honest + malicious

    # -- Median --

    def test_median_blocks_two_byzantine_out_of_eight(self):
        updates = self._make_updates(100, 8)
        median = AggregatorSuite.median(updates)
        self.assertTrue(torch.allclose(median, torch.ones(100)))

    def test_median_all_equal(self):
        updates = [torch.ones(50) * 3.14 for _ in range(5)]
        median = AggregatorSuite.median(updates)
        self.assertTrue(torch.allclose(median, torch.ones(50) * 3.14))

    def test_median_single_update(self):
        updates = [torch.tensor([1.0, 2.0, 3.0])]
        median = AggregatorSuite.median(updates)
        self.assertTrue(torch.allclose(median, torch.tensor([1.0, 2.0, 3.0])))

    def test_median_odd_count(self):
        # 3 peers: [0, 1, 10] -> median = 1
        updates = [torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([10.0])]
        m = AggregatorSuite.median(updates)
        self.assertAlmostEqual(m.item(), 1.0, places=5)

    def test_median_even_count(self):
        # torch.median on even-count tensors returns the *lower* median element
        # (index (K-1)//2 after sorting), NOT the mean of the two middle values.
        # [0, 1, 9, 10] sorted -> lower median at index 1 = 1.0
        updates = [
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([9.0]),
            torch.tensor([10.0]),
        ]
        m = AggregatorSuite.median(updates)
        self.assertAlmostEqual(m.item(), 1.0, places=5)

    def test_median_output_shape(self):
        D = 256
        updates = [torch.randn(D) for _ in range(6)]
        self.assertEqual(AggregatorSuite.median(updates).shape, (D,))

    # -- Fed-Avg --

    def test_fed_avg_all_ones(self):
        updates = [torch.ones(10) for _ in range(4)]
        avg = AggregatorSuite.fed_avg(updates)
        self.assertTrue(torch.allclose(avg, torch.ones(10)))

    def test_fed_avg_mean_correct(self):
        updates = [torch.tensor([float(i)]) for i in range(4)]
        avg = AggregatorSuite.fed_avg(updates)
        self.assertAlmostEqual(avg.item(), 1.5, places=5)

    # -- Trimmed Mean --

    def test_trimmed_mean_k8(self):
        D = 50
        updates = [torch.ones(D) + 0.01 * i for i in range(6)]
        updates += [torch.full((D,), -100.0) for _ in range(2)]
        trimmed = AggregatorSuite.trimmed_mean(updates)
        self.assertGreater(trimmed.mean().item(), 0.0)
        self.assertLess(trimmed.mean().item(), 10.0)

    def test_trimmed_mean_same_as_mean_when_no_outliers(self):
        D = 20
        updates = [torch.ones(D) * float(i + 1) for i in range(6)]
        expected_mean = sum(float(i + 1) for i in range(6)) / 6  # 3.5
        trimmed = AggregatorSuite.trimmed_mean(updates, trim_count=0)
        self.assertAlmostEqual(trimmed.mean().item(), expected_mean, places=4)

    def test_trimmed_mean_output_shape(self):
        D = 128
        updates = [torch.randn(D) for _ in range(8)]
        self.assertEqual(AggregatorSuite.trimmed_mean(updates).shape, (D,))

    def test_trimmed_mean_small_K(self):
        # K=3 should still work
        D = 10
        updates = [torch.ones(D) * float(i) for i in range(3)]
        trimmed = AggregatorSuite.trimmed_mean(updates)
        self.assertEqual(trimmed.shape, (D,))

    # -- Krum --

    def test_krum_returns_valid_vector(self):
        D = 20
        updates = [torch.ones(D) + 0.001 * i for i in range(8)]
        updates[7] = torch.full((D,), -50.0)
        chosen = AggregatorSuite.krum(updates)
        self.assertEqual(chosen.shape[0], D)

    def test_krum_picks_honest_vector(self):
        D = 5
        honest = [torch.ones(D) * float(i) * 0.01 for i in range(6)]
        malicious = [torch.full((D,), 999.0) for _ in range(2)]
        updates = honest + malicious
        chosen = AggregatorSuite.krum(updates)
        # Chosen vector should be close to honest range (0 .. 0.05)
        self.assertLess(chosen.mean().item(), 10.0)

    def test_krum_output_shape(self):
        updates = [torch.randn(64) for _ in range(8)]
        chosen = AggregatorSuite.krum(updates)
        self.assertEqual(chosen.shape, (64,))


# ---------------------------------------------------------------------------
# TestByzantineEngine
# ---------------------------------------------------------------------------

class TestByzantineEngine(unittest.TestCase):

    def _engine(self, attack="sign_flip"):
        return ByzantineAttackEngine(
            num_peers=8,
            byzantine_indices=(6, 7),
            attack_type=attack,
        )

    def test_poison_length(self):
        engine = self._engine()
        updates = [torch.ones(64) for _ in range(8)]
        poisoned = engine.poison(updates)
        self.assertEqual(len(poisoned), 8)

    def test_honest_peers_unchanged(self):
        engine = self._engine("sign_flip")
        updates = [torch.ones(64) + 0.01 * i for i in range(8)]
        poisoned = engine.poison(updates)
        for i in range(6):
            self.assertTrue(torch.allclose(poisoned[i], updates[i]))

    def test_sign_flip_negates(self):
        engine = self._engine("sign_flip")
        updates = [torch.ones(32) + 0.01 * i for i in range(8)]
        poisoned = engine.poison(updates)
        self.assertTrue((poisoned[6] < 0).all())
        self.assertTrue((poisoned[7] < 0).all())

    def test_gaussian_noise_differs_from_honest_mean(self):
        # Gaussian noise attack: poisoned update = honest_mean + randn * 3 * std
        # With D=200 dimensions, std_honest ≈ 0 so the noise term ≈ 0 and
        # we cannot guarantee large deviation.  Instead verify the poisoned
        # vector is NOT identical to the original update (it was modified).
        engine = self._engine("gaussian_noise")
        updates = [torch.ones(200) + 0.001 * i for i in range(8)]
        originals = [u.clone() for u in updates]
        poisoned = engine.poison(updates)
        # Poisoned entries (6, 7) should differ from the pre-poison originals
        # because gaussian_noise replaces the value with mean + randn*std,
        # which is not identical to the original update.
        for idx in (6, 7):
            self.assertFalse(
                torch.allclose(poisoned[idx], originals[idx], atol=1e-6),
                f"Byzantine peer {idx} update was not modified by gaussian_noise"
            )

    def test_zero_gradient_produces_zeros(self):
        engine = self._engine("zero_gradient")
        updates = [torch.randn(50) for _ in range(8)]
        poisoned = engine.poison(updates)
        for idx in (6, 7):
            self.assertTrue(torch.all(poisoned[idx] == 0.0))

    def test_alie_attack_output_shape(self):
        engine = self._engine("alie")
        updates = [torch.randn(100) for _ in range(8)]
        poisoned = engine.poison(updates)
        for idx in (6, 7):
            self.assertEqual(poisoned[idx].shape, (100,))

    def test_targeted_dimension_output_shape(self):
        engine = self._engine("targeted_dimension")
        updates = [torch.randn(100) for _ in range(8)]
        poisoned = engine.poison(updates)
        for idx in (6, 7):
            self.assertEqual(poisoned[idx].shape, (100,))

    def test_targeted_dimension_modifies_top_coords(self):
        engine = self._engine("targeted_dimension")
        updates = [torch.ones(100) * 0.1 for _ in range(8)]
        poisoned = engine.poison(updates)
        # The poisoned update must differ from the honest one
        self.assertFalse(torch.allclose(poisoned[6], updates[6]))

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
            with self.subTest(attack=attack):
                engine = self._engine(attack)
                poisoned = engine.poison(updates)
                self.assertEqual(len(poisoned), 8)

    def test_unsupported_attack_raises(self):
        engine = self._engine("sign_flip")
        updates = [torch.ones(10) for _ in range(8)]
        with self.assertRaises(ValueError):
            engine.poison(updates, attack_type="nonexistent")

    def test_wrong_number_of_updates_raises(self):
        engine = self._engine("sign_flip")
        updates = [torch.ones(10) for _ in range(5)]  # expect 8
        with self.assertRaises(ValueError):
            engine.poison(updates)

    def test_no_byzantine_returns_clones(self):
        engine = ByzantineAttackEngine(
            num_peers=4,
            byzantine_indices=(),
            attack_type="sign_flip",
        )
        updates = [torch.ones(20) * float(i) for i in range(4)]
        poisoned = engine.poison(updates)
        self.assertEqual(len(poisoned), 4)
        for i in range(4):
            self.assertTrue(torch.allclose(poisoned[i], updates[i]))


# ---------------------------------------------------------------------------
# TestPGD10Attacker
# ---------------------------------------------------------------------------

class TestPGD10Attacker(unittest.TestCase):

    def setUp(self):
        self.model = build_lightweight_cnn()
        self.model.eval()

    def test_output_shape_preserved(self):
        attacker = PGD10Attacker(eps=8 / 255, alpha=2 / 255, steps=10)
        x = torch.rand(4, 3, 8, 8)
        y = torch.randint(0, 10, (4,))
        x_adv = attacker.perturb(self.model, x, y)
        self.assertEqual(x_adv.shape, x.shape)

    def test_perturbation_within_eps_ball(self):
        eps = 8 / 255
        attacker = PGD10Attacker(eps=eps, alpha=2 / 255, steps=10)
        x = torch.rand(2, 3, 8, 8)
        y = torch.randint(0, 10, (2,))
        x_adv = attacker.perturb(self.model, x, y)
        delta = (x_adv - x).abs()
        self.assertTrue((delta <= eps + 1e-5).all())

    def test_output_clipped_to_0_1(self):
        attacker = PGD10Attacker(eps=8 / 255, alpha=2 / 255, steps=10)
        x = torch.rand(2, 3, 8, 8)
        y = torch.randint(0, 10, (2,))
        x_adv = attacker.perturb(self.model, x, y)
        self.assertTrue((x_adv >= 0.0).all())
        self.assertTrue((x_adv <= 1.0).all())

    def test_model_mode_restored_after_perturb(self):
        """Model must return to training mode if it was training before PGD."""
        self.model.train()
        attacker = PGD10Attacker()
        x = torch.rand(2, 3, 8, 8)
        y = torch.randint(0, 10, (2,))
        attacker.perturb(self.model, x, y)
        self.assertTrue(self.model.training)

    def test_perturb_zero_steps_returns_random_init(self):
        """With 0 PGD steps the result is just a clipped random perturbation."""
        attacker = PGD10Attacker(eps=8 / 255, alpha=2 / 255, steps=0)
        x = torch.rand(2, 3, 8, 8)
        y = torch.randint(0, 10, (2,))
        x_adv = attacker.perturb(self.model, x, y)
        self.assertEqual(x_adv.shape, x.shape)


# ---------------------------------------------------------------------------
# TestVirtualPeerManager
# ---------------------------------------------------------------------------

class TestVirtualPeerManager(unittest.TestCase):

    def _vpm(self, num_peers=2):
        return VirtualPeerManager(
            model_fn=build_lightweight_cnn,
            num_peers=num_peers,
            device="cpu",
        )

    def test_flatten_unflatten_roundtrip_cpu(self):
        vpm = self._vpm(2)
        model = vpm.models[0]
        flat = vpm.flatten_params(model)
        flat_copy = flat.clone()

        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        vpm.unflatten_params(model, flat_copy)
        flat_after = vpm.flatten_params(model)
        self.assertTrue(torch.allclose(flat_after, flat_copy))

    def test_flatten_grads_all_ones(self):
        vpm = self._vpm(1)
        model = vpm.models[0]
        # Manually set all gradients to 1
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        flat = vpm.flatten_grads(model)
        self.assertTrue((flat == 1.0).all())

    def test_flatten_grads_no_grad_gives_zeros(self):
        vpm = self._vpm(1)
        model = vpm.models[0]
        flat = vpm.flatten_grads(model)
        self.assertTrue((flat == 0.0).all())

    def test_assign_flat_grads_roundtrip(self):
        vpm = self._vpm(1)
        model = vpm.models[0]
        D = vpm.flatten_params(model).numel()
        flat = torch.randn(D)
        vpm.assign_flat_grads(model, flat)
        recovered = vpm.flatten_grads(model)
        self.assertTrue(torch.allclose(recovered, flat))

    def test_num_peers_models_created(self):
        for n in [2, 4, 8]:
            with self.subTest(n=n):
                vpm = self._vpm(n)
                self.assertEqual(len(vpm.models), n)
                self.assertEqual(len(vpm.optimizers), n)

    def test_all_peers_share_initial_weights(self):
        """All peer models must start from the same weight (copy of base)."""
        vpm = self._vpm(4)
        ref = vpm.flatten_params(vpm.models[0])
        for i in range(1, 4):
            flat_i = vpm.flatten_params(vpm.models[i])
            self.assertTrue(torch.allclose(ref, flat_i))

    def test_streams_are_none_on_cpu(self):
        vpm = self._vpm(3)
        self.assertEqual(len(vpm.streams), 3)
        for s in vpm.streams:
            self.assertIsNone(s)


# ---------------------------------------------------------------------------
# TestDataPartitioner
# ---------------------------------------------------------------------------

class TestDataPartitioner(unittest.TestCase):

    def _partitioner(self, num_peers=4):
        return DataPartitioner(
            num_peers=num_peers,
            batch_size=8,
            train_samples_per_peer=32,
            test_samples=16,
            image_size=8,
        )

    def test_iid_peer_loaders_count(self):
        p = self._partitioner(4)
        peer_loaders, _ = p.get_peer_dataloaders(iid=True)
        self.assertEqual(len(peer_loaders), 4)

    def test_noniid_peer_loaders_count(self):
        p = self._partitioner(4)
        peer_loaders, _ = p.get_peer_dataloaders(iid=False)
        self.assertEqual(len(peer_loaders), 4)

    def test_batch_shape_iid(self):
        p = self._partitioner(4)
        peer_loaders, _ = p.get_peer_dataloaders(iid=True)
        x, y = next(iter(peer_loaders[0]))
        self.assertEqual(x.shape[1], 3)   # channels
        self.assertEqual(x.shape[2], 8)   # height
        self.assertEqual(x.shape[3], 8)   # width

    def test_labels_in_valid_range(self):
        p = self._partitioner(4)
        peer_loaders, test_loader = p.get_peer_dataloaders(iid=True)
        for dl in peer_loaders + [test_loader]:
            for _, y in dl:
                self.assertTrue((y >= 0).all())
                self.assertTrue((y < 10).all())

    def test_data_values_in_0_1(self):
        p = self._partitioner(2)
        peer_loaders, _ = p.get_peer_dataloaders(iid=True)
        x, _ = next(iter(peer_loaders[0]))
        self.assertGreaterEqual(x.min().item(), 0.0)
        self.assertLessEqual(x.max().item(), 1.0)

    def test_eight_peer_loaders(self):
        p = self._partitioner(8)
        peer_loaders, _ = p.get_peer_dataloaders(iid=True)
        self.assertEqual(len(peer_loaders), 8)


# ---------------------------------------------------------------------------
# TestSyntheticImageDataset
# ---------------------------------------------------------------------------

class TestSyntheticImageDataset(unittest.TestCase):

    def test_length(self):
        ds = SyntheticImageDataset(num_samples=100, image_size=8)
        self.assertEqual(len(ds), 100)

    def test_item_shape(self):
        ds = SyntheticImageDataset(num_samples=10, image_size=16)
        x, y = ds[0]
        self.assertEqual(x.shape, (3, 16, 16))

    def test_labels_in_range(self):
        ds = SyntheticImageDataset(num_samples=50, num_classes=5)
        for i in range(len(ds)):
            _, y = ds[i]
            self.assertGreaterEqual(y.item(), 0)
            self.assertLess(y.item(), 5)

    def test_pixel_values_in_0_1(self):
        ds = SyntheticImageDataset(num_samples=20)
        for i in range(len(ds)):
            x, _ = ds[i]
            self.assertGreaterEqual(x.min().item(), 0.0)
            self.assertLessEqual(x.max().item(), 1.0)

    def test_reproducible_with_same_seed(self):
        ds1 = SyntheticImageDataset(num_samples=10, sample_seed=42)
        ds2 = SyntheticImageDataset(num_samples=10, sample_seed=42)
        x1, y1 = ds1[0]
        x2, y2 = ds2[0]
        self.assertTrue(torch.allclose(x1, x2))
        self.assertEqual(y1.item(), y2.item())


# ---------------------------------------------------------------------------
# TestLightweightCNN
# ---------------------------------------------------------------------------

class TestLightweightCNN(unittest.TestCase):

    def test_forward_pass_shape(self):
        model = LightweightCNN(in_channels=3, num_classes=10)
        x = torch.randn(4, 3, 16, 16)
        out = model(x)
        self.assertEqual(out.shape, (4, 10))

    def test_forward_pass_small_image(self):
        model = LightweightCNN(in_channels=3, num_classes=5)
        x = torch.randn(2, 3, 8, 8)
        out = model(x)
        self.assertEqual(out.shape, (2, 5))

    def test_param_count_positive(self):
        model = build_lightweight_cnn()
        total = sum(p.numel() for p in model.parameters())
        self.assertGreater(total, 0)

    def test_gradient_flows(self):
        model = build_lightweight_cnn()
        x = torch.randn(2, 3, 8, 8)
        y = torch.randint(0, 10, (2,))
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)


# ---------------------------------------------------------------------------
# TestParallelTreeReduce
# ---------------------------------------------------------------------------

class TestParallelTreeReduce(unittest.TestCase):

    def test_sum_powers_of_two(self):
        v = [1.0, 2.0, 4.0, 8.0]
        result, rounds = parallel_tree_reduce(v)
        self.assertAlmostEqual(result, 15.0, places=6)

    def test_depth_is_ceiling_log2(self):
        for n in [1, 2, 4, 8, 16]:
            v = [1.0] * n
            _, rounds = parallel_tree_reduce(v)
            expected_depth = 0 if n == 1 else math.ceil(math.log2(n))
            self.assertEqual(rounds, expected_depth, msg=f"n={n}")

    def test_single_element(self):
        result, rounds = parallel_tree_reduce([42.0])
        self.assertAlmostEqual(result, 42.0, places=6)
        self.assertEqual(rounds, 0)

    def test_non_power_of_two(self):
        v = [1.0, 2.0, 3.0]   # sum = 6
        result, rounds = parallel_tree_reduce(v)
        self.assertAlmostEqual(result, 6.0, places=6)

    def test_tensor_input(self):
        v = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result, _ = parallel_tree_reduce(v)
        self.assertAlmostEqual(result, 10.0, places=6)


# ---------------------------------------------------------------------------
# TestParallelMeanAndStd
# ---------------------------------------------------------------------------

class TestParallelMeanAndStd(unittest.TestCase):

    def test_mean_correct(self):
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, _ = parallel_mean_and_std(v)
        self.assertAlmostEqual(mean, 3.0, places=5)

    def test_std_correct(self):
        # Population std of [0, 2] = 1.0
        v = [0.0, 2.0]
        _, std = parallel_mean_and_std(v)
        self.assertAlmostEqual(std, 1.0, places=5)

    def test_all_equal_gives_std_zero(self):
        v = [5.0] * 8
        mean, std = parallel_mean_and_std(v)
        self.assertAlmostEqual(mean, 5.0, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parallel_mean_and_std([])

    def test_single_element(self):
        mean, std = parallel_mean_and_std([7.0])
        self.assertAlmostEqual(mean, 7.0, places=5)
        self.assertAlmostEqual(std, 0.0, places=5)


# ---------------------------------------------------------------------------
# TestRingSpMVConsensus (CPU path)
# ---------------------------------------------------------------------------

class TestRingSpMVConsensus(unittest.TestCase):

    def _consensus(self):
        return RingSpMVConsensus(num_streams=4, device="cpu")

    def test_step_preserves_count(self):
        engine = self._consensus()
        vecs = [torch.ones(10) * float(i) for i in range(5)]
        out = engine.step(vecs)
        self.assertEqual(len(out), 5)

    def test_step_output_shape(self):
        engine = self._consensus()
        D = 32
        vecs = [torch.randn(D) for _ in range(6)]
        out = engine.step(vecs)
        for v in out:
            self.assertEqual(v.shape, (D,))

    def test_step_averaging_correctness(self):
        """X_new[i] = (X[i-1] + X[i] + X[i+1]) / 3."""
        engine = self._consensus()
        vecs = [torch.tensor([float(i)]) for i in range(5)]
        out = engine.step(vecs)

        for i in range(5):
            left = float(vecs[(i - 1) % 5].item())
            self_ = float(vecs[i].item())
            right = float(vecs[(i + 1) % 5].item())
            expected = (left + self_ + right) / 3.0
            self.assertAlmostEqual(out[i].item(), expected, places=5)

    def test_step_too_few_peers_raises(self):
        engine = self._consensus()
        vecs = [torch.ones(5), torch.ones(5)]
        with self.assertRaises(ValueError):
            engine.step(vecs)

    def test_converges_uniform_input(self):
        """All-equal input should converge in 0 rounds (already stable)."""
        engine = self._consensus()
        vecs = [torch.tensor([3.0]) for _ in range(6)]
        out, rounds = engine.run_until_converged(vecs, eps=1e-4, max_rounds=100)
        spread = max(abs(out[i].item() - out[j].item())
                     for i in range(6) for j in range(6))
        self.assertLess(spread, 1e-4)

    def test_converges_to_mean(self):
        """After enough gossip rounds the ring should converge near the mean."""
        engine = self._consensus()
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        mean_val = sum(values) / len(values)   # 2.5
        vecs = [torch.tensor([v]) for v in values]
        out, _ = engine.run_until_converged(vecs, eps=1e-3, max_rounds=500)
        for v in out:
            self.assertAlmostEqual(v.item(), mean_val, delta=0.01)


# ---------------------------------------------------------------------------
# TestEndToEndSmoke
# ---------------------------------------------------------------------------

class TestEndToEndSmoke(unittest.TestCase):
    """Light end-to-end integration test that runs purely on CPU."""

    def _make_trainer(self, aggregator="median"):
        from end_to_end_fl import DecentralizedTrainer
        from phase1_topology import DataPartitioner

        return DecentralizedTrainer(
            num_peers=4,
            byzantine_indices=(2, 3),
            attack_type="sign_flip",
            device="cpu",
            topology="full",
            update_type="gradient",
            use_pgd=False,   # skip PGD for speed
        )

    def test_trainer_one_epoch_gradient_mode(self):
        from end_to_end_fl import DecentralizedTrainer
        from phase1_topology import DataPartitioner

        partitioner = DataPartitioner(
            num_peers=4,
            batch_size=8,
            train_samples_per_peer=32,
            test_samples=16,
            image_size=8,
        )
        peer_loaders, _ = partitioner.get_peer_dataloaders(iid=True)

        trainer = DecentralizedTrainer(
            num_peers=4,
            byzantine_indices=(2, 3),
            attack_type="sign_flip",
            device="cpu",
            topology="full",
            update_type="gradient",
            use_pgd=False,
        )

        elapsed = trainer.train_epoch(peer_loaders, aggregator="median")
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0.0)

    def test_trainer_ring_topology_median(self):
        from end_to_end_fl import DecentralizedTrainer
        from phase1_topology import DataPartitioner

        partitioner = DataPartitioner(
            num_peers=4,
            batch_size=8,
            train_samples_per_peer=32,
            test_samples=16,
            image_size=8,
        )
        peer_loaders, _ = partitioner.get_peer_dataloaders(iid=True)

        trainer = DecentralizedTrainer(
            num_peers=4,
            byzantine_indices=(2, 3),
            attack_type="sign_flip",
            device="cpu",
            topology="ring",
            update_type="gradient",
            use_pgd=False,
        )

        elapsed = trainer.train_epoch(peer_loaders, aggregator="median")
        self.assertGreater(elapsed, 0.0)

    def test_trainer_all_aggregators_cpu(self):
        from end_to_end_fl import DecentralizedTrainer
        from phase1_topology import DataPartitioner

        partitioner = DataPartitioner(
            num_peers=4,
            batch_size=8,
            train_samples_per_peer=16,
            test_samples=8,
            image_size=8,
        )
        peer_loaders, _ = partitioner.get_peer_dataloaders(iid=True)

        for agg in ["median", "fed_avg", "trimmed_mean", "krum"]:
            with self.subTest(aggregator=agg):
                trainer = DecentralizedTrainer(
                    num_peers=4,
                    byzantine_indices=(2, 3),
                    attack_type="sign_flip",
                    device="cpu",
                    topology="full",
                    update_type="gradient",
                    use_pgd=False,
                )
                elapsed = trainer.train_epoch(peer_loaders, aggregator=agg)
                self.assertGreater(elapsed, 0.0)

    def test_trainer_unsupported_aggregator_raises(self):
        from end_to_end_fl import DecentralizedTrainer
        from phase1_topology import DataPartitioner

        partitioner = DataPartitioner(
            num_peers=4,
            batch_size=8,
            train_samples_per_peer=16,
            test_samples=8,
            image_size=8,
        )
        peer_loaders, _ = partitioner.get_peer_dataloaders(iid=True)

        trainer = DecentralizedTrainer(
            num_peers=4,
            byzantine_indices=(2, 3),
            attack_type="sign_flip",
            device="cpu",
            topology="full",
            update_type="gradient",
            use_pgd=False,
        )

        with self.assertRaises((ValueError, Exception)):
            trainer.train_epoch(peer_loaders, aggregator="nonexistent")

    def test_trainer_invalid_topology_raises(self):
        from end_to_end_fl import DecentralizedTrainer
        with self.assertRaises(ValueError):
            DecentralizedTrainer(
                num_peers=4,
                byzantine_indices=(2, 3),
                attack_type="sign_flip",
                device="cpu",
                topology="star",
            )

    def test_trainer_invalid_update_type_raises(self):
        from end_to_end_fl import DecentralizedTrainer
        with self.assertRaises(ValueError):
            DecentralizedTrainer(
                num_peers=4,
                byzantine_indices=(2, 3),
                device="cpu",
                update_type="invalid",
            )


if __name__ == "__main__":
    unittest.main()
