#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <vector>
/**
 * TODO1: cuLaunchKernel
 * TODO1: cuMemcpyDtoH
 */

// Error checking macro
#define CUDA_CHECK(err)                                                                                                     \
    if (err != CUDA_SUCCESS) {                                                                                              \
        std::cerr << "CUDA error: " << err << " at line " << __LINE__ << std::endl;                                         \
        exit(EXIT_FAILURE);                                                                                                 \
    }

int main() {
    CUDA_CHECK(cuInit(0));
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0));

    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device));

    // Host memory allocation
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N, 0.0f);

    // Device pointers
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc(&d_A, size);
    std::cout << "complete alloc A" << std::endl;
    cuMemAlloc(&d_B, size);
    std::cout << "complete alloc B" << std::endl;
    cuMemAlloc(&d_C, size);
    std::cout << "complete alloc C" << std::endl;

    // Copy data to device
    CUDA_CHECK(cuMemcpyHtoD(d_A, h_A.data(), size));
    std::cout << "complete transfer A" << std::endl;
    CUDA_CHECK(cuMemcpyHtoD(d_B, h_B.data(), size));
    std::cout << "complete transfer B" << std::endl;

    // Load module and kernel
    CUmodule module;
    CUDA_CHECK(cuModuleLoad(&module, "/root/fan-ege/vector_add.ptx"));
    std::cout << "completing module load" << std::endl;
    CUfunction vectorAdd;
    CUDA_CHECK(cuModuleGetFunction(&vectorAdd, module, "vectorAdd"));
    std::cout << "completing function get" << std::endl;

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    void *args[] = {&d_A, &d_B, &d_C, &N};
    cuLaunchKernel(vectorAdd, blocksPerGrid, 1, 1, // Grid dimensions
                   threadsPerBlock, 1, 1,          // Block dimensions
                   0, nullptr,                     // Shared memory and stream
                   args, nullptr);                 // Kernel arguments

    // Copy result back to host
    printf("before the transfer, h_C[0] = %f\n", h_C[0]);
    CUDA_CHECK(cuMemcpyDtoH(h_C.data(), d_C, size));
    std::cout << "complete transfer C" << std::endl;

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            printf("h_C[%d] = %f\n", i, h_C[i]);
            success = false;
            break;
        }
    }

    std::cout << (success ? "Test PASSED!" : "Test FAILED!") << std::endl;
    printf("A=%p B=%p C=%p\n",(void*)d_A,(void*)d_B,(void*)d_C);

    // Cleanup
    CUDA_CHECK(cuMemFree(d_A));
    CUDA_CHECK(cuMemFree(d_B));
    CUDA_CHECK(cuMemFree(d_C));
    CUDA_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cuCtxDestroy(context));

    std::cout << "successfuly cleanup" << std::endl;
    return 0;
}
