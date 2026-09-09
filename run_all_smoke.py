import torch

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
