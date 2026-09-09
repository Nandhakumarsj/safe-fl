#include <cuda_runtime.h>

__global__ void ring_spmv_kernel(
    const float* __restrict__ X,
    float* __restrict__ X_new,
    int N,
    int D,
    int start,
    int end
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end) {
        int row = idx / D;
        int col = idx - row * D;

        int left = (row == 0) ? (N - 1) : (row - 1);
        int right = (row == N - 1) ? 0 : (row + 1);

        float self_val = X[idx];
        float left_val = X[left * D + col];
        float right_val = X[right * D + col];

        X_new[idx] = (left_val + self_val + right_val) / 3.0f;
    }
}

extern "C" void launch_ring_spmv(
    const float* X_ptr,
    float* X_new_ptr,
    int N,
    int D,
    int num_streams,
    const cudaStream_t* streams
) {
    const int threads = 256;
    int total = N * D;

    int chunk = (total + num_streams - 1) / num_streams;

    for (int sid = 0; sid < num_streams; ++sid) {
        int start = sid * chunk;
        if (start >= total) {
            break;
        }

        int remaining = total - start;
        int count = (chunk < remaining) ? chunk : remaining;
        int end = start + count;
        int blocks = (count + threads - 1) / threads;

        ring_spmv_kernel<<<blocks, threads, 0, streams[sid]>>>(
            X_ptr,
            X_new_ptr,
            N,
            D,
            start,
            end
        );
    }
}
