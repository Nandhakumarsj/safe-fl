import numpy as np
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
