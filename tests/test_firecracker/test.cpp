// firecracker_test.cpp — instrumented without introducing any extra CUDA calls
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

static inline std::string hex64(uint64_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << v << std::dec;
    return oss.str();
}

// Minimal check macro (no cuGetErrorString/Name — prints numeric status)
#define CUDA_DO(call) do {                                     \
    CUresult _st = (call);                                     \
    std::cout << #call << " -> " << (int)_st;                  \
    if (_st != CUDA_SUCCESS) {                                 \
        std::cout << "  [ERROR at " << __FILE__ << ":" << __LINE__ << "]\n"; \
        std::exit(1);                                          \
    } else {                                                   \
        std::cout << "  [OK]\n";                               \
    }                                                          \
} while(0)

int main(int argc, char** argv) {
    // args: <path_to_ptx/cubin> [kernel_name] [N]
    const char* module_path = (argc >= 2) ? argv[1] : "vector_add.ptx";
    const char* kernel_name = (argc >= 3) ? argv[2] : "vector_add";
    size_t N = (argc >= 4) ? static_cast<size_t>(std::strtoull(argv[3], nullptr, 10)) : (1ull << 20);

    std::cout << "=== microvm cuda remoting test ===\n";
    std::cout << "module_path: " << module_path << "\n";
    std::cout << "kernel_name: " << kernel_name << "\n";
    std::cout << "N: " << N << " (" << (N * sizeof(float)) << " bytes per buffer)\n";

    // Init / device / context
    CUDA_DO(cuInit(0));
    CUdevice device;
    CUDA_DO(cuDeviceGet(&device, 0));
    std::cout << "device (index 0) handle: " << device << "\n";

    CUcontext context = nullptr;
    CUDA_DO(cuCtxCreate(&context, 0, device));
    std::cout << "context: " << context << "\n";

    // Host buffers
    const size_t bytes = N * sizeof(float);
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N, 0.0f);
    std::cout << "host init: A[0]=" << h_A[0] << " B[0]=" << h_B[0] << " C[0]=" << h_C[0] << "\n";

    // Device buffers
    CUdeviceptr d_A = 0, d_B = 0, d_C = 0;
    CUDA_DO(cuMemAlloc(&d_A, bytes));
    std::cout << "d_A = " << hex64(static_cast<uint64_t>(d_A)) << " size=" << bytes << "\n";
    CUDA_DO(cuMemAlloc(&d_B, bytes));
    std::cout << "d_B = " << hex64(static_cast<uint64_t>(d_B)) << " size=" << bytes << "\n";
    CUDA_DO(cuMemAlloc(&d_C, bytes));
    std::cout << "d_C = " << hex64(static_cast<uint64_t>(d_C)) << " size=" << bytes << "\n";

    if (d_A == d_B || d_A == d_C || d_B == d_C) {
        std::cout << "FATAL: duplicate CUdeviceptr detected (aliasing)\n";
        std::cout << "d_A=" << hex64(d_A) << " d_B=" << hex64(d_B) << " d_C=" << hex64(d_C) << "\n";
        std::exit(2);
    }

    // HtoD
    CUDA_DO(cuMemcpyHtoD(d_A, h_A.data(), bytes));
    std::cout << "HtoD A done; sample host A[0]=" << h_A[0] << "\n";
    CUDA_DO(cuMemcpyHtoD(d_B, h_B.data(), bytes));
    std::cout << "HtoD B done; sample host B[0]=" << h_B[0] << "\n";

    // Module & function
    CUmodule module = nullptr;
    CUDA_DO(cuModuleLoad(&module, module_path));
    std::cout << "module handle: " << module << "\n";

    CUfunction kernel = nullptr;
    CUDA_DO(cuModuleGetFunction(&kernel, module, kernel_name));
    std::cout << "kernel func: " << kernel << "\n";

    // Launch config
    const unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((N + threads - 1) / threads);
    std::cout << "launch: blocks=" << blocks << " threads=" << threads << "\n";

    // Kernel params (log both values and the addresses of the parameter variables)
    unsigned int N_u32 = static_cast<unsigned int>(N);
    void* params[] = { &d_A, &d_B, &d_C, &N_u32 };

    std::cout << "kernel param VALUES:\n";
    std::cout << "  d_A = " << hex64(static_cast<uint64_t>(d_A)) << "\n";
    std::cout << "  d_B = " << hex64(static_cast<uint64_t>(d_B)) << "\n";
    std::cout << "  d_C = " << hex64(static_cast<uint64_t>(d_C)) << "\n";
    std::cout << "  N   = " << N_u32 << "\n";

    std::cout << "kernel param ADDRESSES (in guest):\n";
    std::cout << "  &d_A = " << hex64(reinterpret_cast<uint64_t>(&d_A)) << "\n";
    std::cout << "  &d_B = " << hex64(reinterpret_cast<uint64_t>(&d_B)) << "\n";
    std::cout << "  &d_C = " << hex64(reinterpret_cast<uint64_t>(&d_C)) << "\n";
    std::cout << "  &N   = " << hex64(reinterpret_cast<uint64_t>(&N_u32)) << "\n";

    // Launch
    CUDA_DO(cuLaunchKernel(
        kernel,
        blocks, 1, 1,
        threads, 1, 1,
        0,              // sharedMemBytes
        nullptr,        // stream
        params,
        nullptr         // extra
    ));
    std::cout << "launched kernel\n";

    // DtoH
    std::cout << "before DtoH, h_C[0] = " << h_C[0] << "\n";
    CUDA_DO(cuMemcpyDtoH(h_C.data(), d_C, bytes));
    std::cout << "after  DtoH, h_C[0] = " << h_C[0] << "\n";

    // Validate compactly (no extra CUDA calls)
    size_t mismatches = 0;
    for (size_t i = 0; i < N; ++i) {
        const float expect = h_A[i] + h_B[i]; // 3.0f
        if (h_C[i] != expect) {
            if (mismatches < 16) {
                std::cout << "mismatch @" << i << ": got " << h_C[i] << " expect " << expect << "\n";
            }
            ++mismatches;
        }
    }
    std::cout << "validation: mismatches=" << mismatches << " of " << N << "\n";
    std::cout << (mismatches == 0 ? "Test PASSED!" : "Test FAILED!") << "\n";

    // Cleanup
    CUDA_DO(cuMemFree(d_A));
    CUDA_DO(cuMemFree(d_B));
    CUDA_DO(cuMemFree(d_C));
    CUDA_DO(cuModuleUnload(module));
    CUDA_DO(cuCtxDestroy(context));
    std::cout << "successfully cleanup\n";
    return (mismatches == 0) ? 0 : 1;
}
