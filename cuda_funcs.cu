#include <cuda_runtime.h>

// Kernel to compute the determinant of 3x3 matrices
__global__ void determinantKernel(float* matrices, float* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float* matrix = &matrices[idx * 9]; // 3x3 matrix
        // Calculate determinant using the rule of Sarrus
        results[idx] = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
                       matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
                       matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    }
}

// Host function to launch the kernel
extern "C" __declspec(dllexport) void computeDeterminants(float* matrices, float* results, int n) {
    float* d_matrices;
    float* d_results;

    // Allocate device memory
    cudaMalloc((void**)&d_matrices, n * 9 * sizeof(float));
    cudaMalloc((void**)&d_results, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_matrices, matrices, n * 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel with one thread per matrix
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    determinantKernel<<<numBlocks, blockSize>>>(d_matrices, d_results, n);

    // Copy results back to host
    cudaMemcpy(results, d_results, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrices);
    cudaFree(d_results);
}

// #include <cuda_runtime.h>
// #include <stdio.h>

// Tolerance for zero check
#define EPSILON 1e-6

// Utility function to check if a float value is approximately zero
__device__ bool is_zero(float value) {
    return fabs(value) < EPSILON;
}

// CUDA Kernel to compute line-plane intersections
__global__ void LinePlaneCrossPointKernel(
    const float* origins, const float* directions,
    const float* p1s, const float* p2s, const float* p3s,
    float* cross_points, bool* has_intersections, int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        // Load line origin and direction
        float origin[3] = { origins[3 * idx], origins[3 * idx + 1], origins[3 * idx + 2] };
        float direction[3] = { directions[3 * idx], directions[3 * idx + 1], directions[3 * idx + 2] };

        // Load plane points
        float p1[3] = { p1s[3 * idx], p1s[3 * idx + 1], p1s[3 * idx + 2] };
        float p2[3] = { p2s[3 * idx], p2s[3 * idx + 1], p2s[3 * idx + 2] };
        float p3[3] = { p3s[3 * idx], p3s[3 * idx + 1], p3s[3 * idx + 2] };

        // Calculate vectors v1 and v2 from the plane points
        float v1[3] = { p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
        float v2[3] = { p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2] };

        // Calculate the normal vector n to the plane using the cross product
        float n[3] = {
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        };

        // Check if the line is parallel to the plane (dot product == 0)
        float dot_nd = n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2];
        if (is_zero(dot_nd)) {
            has_intersections[idx] = false;
            return;
        }

        // Calculate parameter t for the line-plane intersection
        float t = ((p1[0] - origin[0]) * n[0] + (p1[1] - origin[1]) * n[1] + (p1[2] - origin[2]) * n[2]) / dot_nd;

        // Calculate the intersection point
        cross_points[3 * idx] = origin[0] + t * direction[0];
        cross_points[3 * idx + 1] = origin[1] + t * direction[1];
        cross_points[3 * idx + 2] = origin[2] + t * direction[2];
        has_intersections[idx] = true;
    }
}

extern "C" __declspec(dllexport) void LinePlaneCrossPoints(
    float* origins, float* directions,
    float* p1s, float* p2s, float* p3s,
    float* cross_points, bool* has_intersections, int num_elements) {
    
    float *d_origins, *d_directions, *d_p1s, *d_p2s, *d_p3s, *d_cross_points;
    bool* d_has_intersections;

    // Allocate memory on the device (GPU)
    cudaMalloc((void**)&d_origins, 3 * num_elements * sizeof(float));
    cudaMalloc((void**)&d_directions, 3 * num_elements * sizeof(float));
    cudaMalloc((void**)&d_p1s, 3 * num_elements * sizeof(float));
    cudaMalloc((void**)&d_p2s, 3 * num_elements * sizeof(float));
    cudaMalloc((void**)&d_p3s, 3 * num_elements * sizeof(float));
    cudaMalloc((void**)&d_cross_points, 3 * num_elements * sizeof(float));
    cudaMalloc((void**)&d_has_intersections, num_elements * sizeof(bool));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_origins, origins, 3 * num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_directions, directions, 3 * num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1s, p1s, 3 * num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2s, p2s, 3 * num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3s, p3s, 3 * num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel with computed grid and block dimensions
    LinePlaneCrossPointKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_origins, d_directions, d_p1s, d_p2s, d_p3s, d_cross_points, d_has_intersections, num_elements);

    // Copy the result back from the device to the host
    cudaMemcpy(cross_points, d_cross_points, 3 * num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(has_intersections, d_has_intersections, num_elements * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_origins);
    cudaFree(d_directions);
    cudaFree(d_p1s);
    cudaFree(d_p2s);
    cudaFree(d_p3s);
    cudaFree(d_cross_points);
    cudaFree(d_has_intersections);
}