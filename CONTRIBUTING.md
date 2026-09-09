# Contributing to Multi-Stream CUDA Decentralized Federated Learning

Thank you for your interest in contributing to the **Multi-Stream CUDA Decentralized Federated Learning** project! We welcome contributions ranging from bug reports and documentation enhancements to novel Byzantine consensus kernels and adversarial defense strategies.

---

## 1. Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

## 2. Getting Started & Development Setup

### Option A: Local Development Setup
1. **Prerequisites**:
   - Python 3.9+
   - NVIDIA GPU with compute capability $\ge 7.0$ (Turing, Ampere, Ada Lovelace, Hopper)
   - NVIDIA CUDA Toolkit $\ge 12.0$ with `nvcc` in your `PATH`
   - C++ Host Compiler:
     - **Linux**: `gcc` / `g++` $\ge 9$
     - **Windows**: Microsoft Visual Studio C++ Build Tools (`cl.exe` in `PATH`)
   - `ninja-build` (`pip install ninja`)

2. **Clone & Install**:
   ```bash
   git clone https://github.com/Nandhakumarsj/safe-fl.git
   cd safe-fl
   pip install -r requirements.txt
   pip install ninja python-docx
   ```

3. **Build CUDA Extensions In-Place (Optional for JIT compilation)**:
   ```bash
   python setup.py build_ext --inplace
   ```

### Option B: Docker Container Setup (Recommended)
To eliminate compiler and dependency configuration mismatches:
```bash
# Build and run the smoke test in Docker
docker compose up --build

# Or enter an interactive container shell
docker compose run --rm dfl-cuda bash
```

---

## 3. Verification & Testing Workflow

Always ensure existing tests pass before submitting changes:

```bash
# 1. Run Unit Tests (Topology, Byzantine attacks, Aggregators, Flattening)
python -m unittest discover tests

# 2. Run Single-Epoch Smoke Verification
python run_all_smoke.py

# 3. Run Benchmark Matrix Across Attacks & Defenses
python benchmark_aggregators.py

# 4. Run Publication Evaluation & Plotting
python evaluate_and_plot.py
```

---

## 4. Coding Standards & Parallel Guidelines

- **Zero Warp Divergence in CUDA Kernels**:
  Any coordinate-wise reduction or sorting kernel must remain data-oblivious. Use Compare-and-Swap (CAS) sorting networks using `fminf`/`fmaxf` rather than branching `if-else` blocks.
- **Register-First Storage**:
  Small working sets ($K \le 8$) should be held strictly in registers (`float v0..v7`) rather than shared memory (`__shared__`) to eliminate bank conflicts and latency overheads.
- **Asynchronous CUDA Streams**:
  All per-peer operations must be enqueued inside their assigned CUDA stream:
  ```python
  with torch.cuda.stream(stream):
      # peer computation
  ```
  Never place a blocking `torch.cuda.synchronize()` inside the per-peer loop; defer synchronization to the phase boundaries.
- **Disjoint Memory Access (EREW Discipline)**:
  Ensure no two threads or streams write to overlapping output memory addresses.

---

## 5. Submitting Pull Requests

1. Fork the repository and create your branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Commit your changes using conventional commit messages:
   - `feat(scope): description`
   - `fix(scope): description`
   - `docs(scope): description`
   - `perf(scope): description`
   - `test(scope): description`
3. Ensure all tests and smoke scripts pass.
4. Push your branch and open a Pull Request with a clear description of the problem solved and performance/correctness evidence.
