import time
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
