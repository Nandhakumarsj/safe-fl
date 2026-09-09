#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>

extern "C" void launch_coordinate_wise_median(
    const float* gradients_ptr,
    float* aggregated_ptr,
    int num_coordinates,
    int num_clients,
    int num_streams,
    const cudaStream_t* streams
);

torch::Tensor coordinate_wise_median(
    torch::Tensor gradients,
    std::vector<int64_t> stream_ptrs
) {
    TORCH_CHECK(gradients.dim() == 2, "gradients must be a 2D tensor [D, K]");
    TORCH_CHECK(gradients.dtype() == torch::kFloat32, "gradients must be float32");
    TORCH_CHECK(gradients.is_cuda(), "gradients must be on a CUDA device");
    TORCH_CHECK(
        gradients.is_contiguous(),
        "gradients must be contiguous (call .contiguous() after transpose)"
    );

    int64_t D = gradients.size(0);
    int64_t K = gradients.size(1);

    TORCH_CHECK(
        K == 3 || K == 8,
        "num_clients (K) must be 3 (ring neighbourhood) or 8 (global), got ",
        K
    );

    int num_streams = static_cast<int>(stream_ptrs.size());
    TORCH_CHECK(num_streams > 0, "at least one CUDA stream handle must be supplied");

    at::cuda::CUDAGuard device_guard(gradients.device().index());

    torch::Tensor aggregated = torch::empty({D}, gradients.options());

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        streams[i] = reinterpret_cast<cudaStream_t>(stream_ptrs[i]);
    }

    launch_coordinate_wise_median(
        gradients.data_ptr<float>(),
        aggregated.data_ptr<float>(),
        static_cast<int>(D),
        static_cast<int>(K),
        num_streams,
        streams.data()
    );

    C10_CUDA_CHECK(cudaGetLastError());

    return aggregated;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "coordinate_wise_median",
        &coordinate_wise_median,
        "Coordinate-wise median aggregation over K=3 or K=8 neighbours (CUDA)",
        py::arg("gradients"),
        py::arg("stream_ptrs")
    );
}
