import torch

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
