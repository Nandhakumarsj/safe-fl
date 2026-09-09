# Security Policy

## 1. Supported Versions

We provide security updates and patches for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## 2. Threat Model & Scope

This project implements a decentralized federated learning pipeline hardened against **two simultaneous, independent threat vectors**:

1. **Input-Level Adversarial Threat**:
   - Adversarial sample craft at inference/training time via multi-step gradient perturbation (e.g., $L_\infty$-bounded PGD-10 with budget $\epsilon = 8/255$).
   - Defended locally via stream-parallel adversarial training.

2. **Network-Level Byzantine Threat**:
   - A minority fraction ($f < N/2$) of participating nodes injecting arbitrary corrupted, inverted, or crafted gradients during gossip consensus rounds (e.g., sign-flip, Gaussian noise, zero-gradient, ALIE, and targeted-dimension poisoning).
   - Defended via custom CUDA coordinate-wise median sorting networks and outlier-trimmed aggregators.

### Out of Scope
- Physical GPU side-channel attacks.
- Attacks where Byzantine actors control a majority of nodes ($f \ge N/2$), which exceeds the theoretical breakdown threshold of order-statistic estimators.
- Denial of Service (DoS) attacks on the host operating system.

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability (such as a memory safety issue in custom CUDA bindings, race conditions in multi-stream synchronization, or an algorithmic bypass of the Byzantine consensus filter), please do **NOT** open a public GitHub issue.

Instead, please report the vulnerability privately:

1. Send an encrypted or confidential email to `security@nvidia-parallel.local` or contact repository maintainers directly.
2. Please include:
   - Description of the vulnerability.
   - Minimal reproducible code or attack script.
   - Affected components (`consensus_kernel.cu`, `phase2_security.py`, etc.).
   - Potential impact and mitigation recommendations.

### Response Timetable
- **Acknowledgment**: Within 48 hours of initial report.
- **Assessment & Reproduction**: Within 7 business days.
- **Patch & Advisory Release**: Coordinated with the reporter before public disclosure.
