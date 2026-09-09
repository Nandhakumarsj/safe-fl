import os
import sys
from setuptools import setup, find_packages

# Determine if CUDA extensions can be compiled
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    CUDA_AVAILABLE = torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1"
except ImportError:
    CUDA_AVAILABLE = False
    BuildExtension = None
    CUDAExtension = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ext_modules = []
cmdclass = {}

if CUDA_AVAILABLE and CUDAExtension is not None:
    nvcc_flags = ["-O3", "--use_fast_math"]
    cxx_flags = ["-O3"]
    if os.name == "nt":
        cxx_flags = ["/O2"]

    ext_modules = [
        CUDAExtension(
            name="consensus_cuda",
            sources=[
                os.path.join(_THIS_DIR, "consensus_binding.cpp"),
                os.path.join(_THIS_DIR, "consensus_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
        CUDAExtension(
            name="ring_spmv_cuda",
            sources=[
                os.path.join(_THIS_DIR, "ring_spmv_binding.cpp"),
                os.path.join(_THIS_DIR, "ring_spmv_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="multi_stream_cuda_dfl",
    version="1.0.0",
    description="Multi-Stream CUDA-Accelerated Decentralized Federated Learning Under Dual Threat Security Models",
    author="Nvidia-Parallel Developers",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
)
