#include <cuda_runtime.h>

__device__ __forceinline__ float median3(float a, float b, float c) {
    float mn = fminf(a, fminf(b, c));
    float mx = fmaxf(a, fmaxf(b, c));
    return a + b + c - mn - mx;
}

__device__ __forceinline__ void sort8(
    float &v0, float &v1, float &v2, float &v3,
    float &v4, float &v5, float &v6, float &v7
) {
#define MINMAX(a, b) \
    { \
        float t = a; \
        a = fminf(a, b); \
        b = fmaxf(t, b); \
    }

    MINMAX(v0, v1); MINMAX(v2, v3); MINMAX(v4, v5); MINMAX(v6, v7);
    MINMAX(v0, v2); MINMAX(v1, v3); MINMAX(v4, v6); MINMAX(v5, v7);
    MINMAX(v1, v2); MINMAX(v5, v6); MINMAX(v0, v4); MINMAX(v3, v7);
    MINMAX(v1, v5); MINMAX(v2, v6);
    MINMAX(v1, v4); MINMAX(v3, v6);
    MINMAX(v2, v4); MINMAX(v3, v5);
    MINMAX(v3, v4);

#undef MINMAX
}

__global__ void median3_kernel(
    const float* __restrict__ gradients,
    float* __restrict__ aggregated,
    int start,
    int end
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end) {
        const float* row = gradients + idx * 3;
        aggregated[idx] = median3(row[0], row[1], row[2]);
    }
}

__global__ void median8_kernel(
    const float* __restrict__ gradients,
    float* __restrict__ aggregated,
    int start,
    int end
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end) {
        const float* row = gradients + idx * 8;

        float v0 = row[0];
        float v1 = row[1];
        float v2 = row[2];
        float v3 = row[3];
        float v4 = row[4];
        float v5 = row[5];
        float v6 = row[6];
        float v7 = row[7];

        sort8(v0, v1, v2, v3, v4, v5, v6, v7);

        aggregated[idx] = 0.5f * (v3 + v4);
    }
}

extern "C" void launch_coordinate_wise_median(
    const float* gradients_ptr,
    float* aggregated_ptr,
    int num_coordinates,
    int num_clients,
    int num_streams,
    const cudaStream_t* streams
) {
    const int threads = 256;
    int total = num_coordinates;

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

        if (num_clients == 3) {
            median3_kernel<<<blocks, threads, 0, streams[sid]>>>(
                gradients_ptr,
                aggregated_ptr,
                start,
                end
            );
        } else if (num_clients == 8) {
            median8_kernel<<<blocks, threads, 0, streams[sid]>>>(
                gradients_ptr,
                aggregated_ptr,
                start,
                end
            );
        }
    }
}
