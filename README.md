Multi-Stream CUDA-Accelerated Decentralized Federated Learning Under Dual Threat Security Models

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
