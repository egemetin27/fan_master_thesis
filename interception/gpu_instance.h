#ifndef GPU_INSTANCE_H
#define GPU_INSTANCE_H

#include <cuda.h>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>

#include <functional>
#include <memory>

#include "stream.h"
#include "type_decl.h"

#include <cstdlib> // for std::getenv

#define SHM_PATH "/dev/shm/shared_mem"
#define SHM_SIZE (1 << 20) * sizeof(float)
#define ADD_SYMBOL(symbol)                                                     \
  {                                                                            \
    SYMBOL_TO_STR(symbol), [this]() { CUDA_API_IMPL(symbol) }                  \
  }
/* generate the corresponding cuda function with arguments stored in shared
 * memory. */
#define CUDA_API_IMPL(symbol)                                                  \
  std::cout << "<<<<<<<<<<implemented as " << #symbol << std::endl;            \
  using param_t = FunctionTraits<decltype(symbol)>::ParameterTuple;            \
  param_t args;                                                                \
  deserializer_ >> args;                                                       \
  CUresult result = std::apply(symbol, args);                                  \
  response_.set_curesult(result);                                              \
  response_ << args;

// Small helpers so we don't have to change any class signatures.
static inline const char *fc_shared_mem_path() {
  if (const char *p = std::getenv("FC_SHARED_MEM"))
    return p;
  return "/dev/shm/shared_mem";
}
static inline const char *fc_cuda_pin_path() {
  if (const char *p = std::getenv("FC_CUDA_PIN"))
    return p;
  return "/dev/shm/cuda_pin";
}

/**
 * @class GPUinstance
 * @brief virtual gpu instance to be dedicated to a requesting
 *        microvm.
 * @details handles command recv and cuda respond respond. There is
 *          also a virtual gpu used for store data from the guest os
 *          as host data to be transfered to the gpu.
 */
class GPUInstance {
public:
  explicit GPUInstance(int device_id, int client_fd)
      : client_fd_(client_fd), vgpu_(fc_shared_mem_path(), (1 << 20) * 9),
        vgpu_ptr_(vgpu_.get()),
        pinned_memory_(fc_cuda_pin_path(), SHM_SIZE * 5),
        client_pin_start_(nullptr) {
    cuDeviceGet(&device_, device_id);
    cuCtxCreate(&cucontext_, 0, device_);
  }
  ~GPUInstance() {
    cuCtxDestroy(cucontext_);
    for (auto p : registered_)
      cuMemHostUnregister(p);
    if (client_fd_ != -1)
      printf("the client fd is closed gracefully\n"), close(client_fd_);
  };

  void handle() {
    while (client_fd_ != -1) {
      implement_cuda_function();
    }
  }

  /******************************************************
   *        host CUDA allocated pointers storage        *
   ******************************************************/
private:
  std::vector<std::unique_ptr<CUdeviceptr>> device_ptrs_;
  std::vector<std::unique_ptr<CUmodule>> cumodules_;
  std::vector<std::unique_ptr<CUfunction>> cufuncs_;
  std::vector<std::unique_ptr<CUdevice>> cudevices_;
  std::vector<std::unique_ptr<CUcontext>> cuctxs_;
  std::vector<std::unique_ptr<void *>> hdata_ptrs_;

  CUdevice device_;
  CUcontext cucontext_;

private:
  int client_fd_;
  Response response_;
  void return_result() {
    if (client_fd_ == -1) {
      std::cout << "Skipping result return - client disconnected" << std::endl;
      return;
    }
    ssize_t sent = send(client_fd_, response_.data(), response_.size(), 0);
    if (sent == -1) {
      std::cout << "Failed to send response - client may have disconnected"
                << std::endl;
      close(client_fd_);
      client_fd_ = -1;
    }
  }
  VirtualGPU vgpu_;
  void *vgpu_ptr_;
  PinnedMemory pinned_memory_;
  void *client_pin_start_;

  std::vector<void *> registered_;
  bool is_pinned(void *ptr) {
    return (uintptr_t)ptr >= (uintptr_t)client_pin_start_ &&
           (uintptr_t)ptr <= (uintptr_t)client_pin_start_ + SHM_SIZE * 5;
  }
  void *convert_pin(void *ptr) {
    auto offset = (uintptr_t)ptr - (uintptr_t)client_pin_start_;
    return (char *)pinned_memory_.get() + offset;
  }

private:
  Deserializer deserializer_;
  char *pull_command() {
    deserializer_.clean();
    int bytes_read =
        read(client_fd_, deserializer_.data(), deserializer_.size());
    if (bytes_read <= 0) {
      // Client disconnected or error occurred
      std::cout << "Client disconnected (bytes_read=" << bytes_read
                << "), closing connection" << std::endl;
      close(client_fd_);
      client_fd_ = -1;
      return nullptr;
    }
    char *func_name;
    deserializer_ >> func_name;
    return func_name;
  }

  void implement_cuda_function();
  void resolve_cuda_error(CUresult &result) {
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::cout << "the error should be " << name << std::endl;
  }

private:
  std::unordered_map<std::string, std::function<void()>> func_map = {
      ADD_SYMBOL(cuInit),          ADD_SYMBOL(cuDevicePrimaryCtxRelease),
      ADD_SYMBOL(cuCtxSetCurrent), ADD_SYMBOL(cuCtxPushCurrent),
      ADD_SYMBOL(cuLibraryUnload), ADD_SYMBOL(cuMemsetD8Async),
      ADD_SYMBOL(cuEventRecord),   ADD_SYMBOL(cuStreamSynchronize),
      ADD_SYMBOL(cuCtxDestroy),    ADD_SYMBOL(cuMemFree),
      ADD_SYMBOL(cuModuleUnload),
  };
};

#endif