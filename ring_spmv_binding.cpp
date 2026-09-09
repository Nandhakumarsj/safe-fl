#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void launch_ring_spmv(
    const float* X_ptr,
    float* X_new_ptr,
    int N,
    int D,
    int num_streams,
    const cudaStream_t* streams
);

torch::Tensor ring_spmv(
    torch::Tensor X,
    std::vector<int64_t> stream_ptrs
) {
    TORCH_CHECK(X.dim() == 2, "X must be a 2D tensor [N, D]");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(X.is_cuda(), "X must be on a CUDA device");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    int64_t N = X.size(0);
    int64_t D = X.size(1);

    TORCH_CHECK(N >= 3, "ring topology requires at least 3 peers, got ", N);

    int num_streams = static_cast<int>(stream_ptrs.size());
    TORCH_CHECK(num_streams > 0, "at least one CUDA stream handle must be supplied");

    at::cuda::CUDAGuard device_guard(X.device().index());

    torch::Tensor X_new = torch::empty_like(X);

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        streams[i] = reinterpret_cast<cudaStream_t>(stream_ptrs[i]);
    }

    launch_ring_spmv(
        X.data_ptr<float>(),
        X_new.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(D),
        num_streams,
        streams.data()
    );

    C10_CUDA_CHECK(cudaGetLastError());

    return X_new;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ring_spmv",
        &ring_spmv,
        "One round of ring-topology linear gossip consensus X_new = W.X (CUDA)",
        py::arg("X"),
        py::arg("stream_ptrs")
    );
}
