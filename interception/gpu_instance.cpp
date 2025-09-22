#include "gpu_instance.h"
#include <cuda_runtime.h>
#include <filesystem>
#define FUNC_COMP(func_name, symbol)                                           \
  strcmp(func_name, SYMBOL_TO_STR(symbol)) == 0

//======================================================================================//
void GPUInstance::implement_cuda_function() {
  // cuCtxSetCurrent(cucontext_);
  char *function_name = pull_command();
  if (function_name == nullptr) {
    // Client disconnected, exit function
    return;
  }
  std::cout << "Received GPU commands: " << function_name << std::endl;
  response_.reset();
  //! change the buffer size here (how about transfer the size in the
  //! beginning?)
  // TODO client name should be used in the future
  // HACK use FUNC_MAP
  //  // if(function_name == "cuDeviceGet"){
  //  //     cuDeviceGet(&device, 0);
  //  // }

  // // if(function_name == "cuCtxCreate"){
  // //     cuCtxCreate(&cucontext, 0, device);
  // // }
  if (FUNC_COMP(function_name, cuDeviceGet)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;
    using param_t = FunctionTraits<decltype(cuDeviceGet)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;

    CUdevice device;
    std::get<0>(args) = &device;
    CUresult result = std::apply(cuDeviceGet, args);
    response_.set_curesult(result);
    response_ << device << std::get<1>(args);
  } else if (FUNC_COMP(function_name, cuCtxCreate)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;
    using param_t = FunctionTraits<decltype(cuCtxCreate)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;

    cuctxs_.push_back(std::make_unique<CUcontext>());
    std::get<0>(args) = cuctxs_.back().get();
    CUresult result = std::apply(cuCtxCreate, args);
    response_.set_curesult(result);
    response_ << cuctxs_.back().get();
  } else if (FUNC_COMP(function_name, cuDeviceGetCount)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;
    using param_t = FunctionTraits<decltype(cuDeviceGetCount)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;

    int count;
    std::get<0>(args) = &count;
    CUresult result = std::apply(cuDeviceGetCount, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuDeviceTotalMem)) {
    std::cout << "<<<<<<<<<<implemented as cuDeviceTotalMem" << std::endl;

    using param_t = FunctionTraits<decltype(cuDeviceTotalMem)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;

    // create local for size_t*
    size_t bytes;
    std::get<0>(args) = &bytes;

    CUresult result = std::apply(cuDeviceTotalMem, args);

    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuDeviceGetAttribute)) {
    std::cout << "<<<<<<<<<<implemented as cuDeviceGetAttribute" << std::endl;
    using param_t =
        FunctionTraits<decltype(cuDeviceGetAttribute)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    int pi;
    std::get<0>(args) = &pi;
    CUresult result = std::apply(cuDeviceGetAttribute, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuDriverGetVersion)) {
    std::cout << "<<<<<<<<<<implemented as cuDriverGetVersion" << std::endl;
    using param_t =
        FunctionTraits<decltype(cuDriverGetVersion)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    int version;
    std::get<0>(args) = &version;
    CUresult result = std::apply(cuDriverGetVersion, args);
    response_.set_curesult(result);
    response_ << args;

    resolve_cuda_error(result);
    std::cout << "driver version is " << version << std::endl;
  } else if (FUNC_COMP(function_name, cuDeviceGetUuid)) {
    std::cout << "<<<<<<<<<<implemented as cuDeviceGetUuid" << std::endl;
    using param_t = FunctionTraits<decltype(cuDeviceGetUuid)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUuuid uuid;
    std::get<0>(args) = &uuid;
    CUresult result = std::apply(cuDeviceGetUuid, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuDevicePrimaryCtxRetain)) {
    std::cout << "<<<<<<<<<<implemented as cuDevicePrimaryCtxRetain"
              << std::endl;
    using param_t =
        FunctionTraits<decltype(cuDevicePrimaryCtxRetain)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUcontext ctx;
    std::get<0>(args) = &ctx;
    CUresult result = std::apply(cuDevicePrimaryCtxRetain, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuCtxGetCurrent)) {
    std::cout << "<<<<<<<<<<implemented as cuCtxGetCurrent" << std::endl;
    using param_t = FunctionTraits<decltype(cuCtxGetCurrent)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUcontext ctx;
    std::get<0>(args) = &ctx;
    CUresult result = std::apply(cuCtxGetCurrent, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuCtxGetDevice)) {
    std::cout << "<<<<<<<<<<implemented as cuCtxGetDevice" << std::endl;
    using param_t = FunctionTraits<decltype(cuCtxGetDevice)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUdevice dev;
    std::get<0>(args) = &dev;
    CUresult result = std::apply(cuCtxGetDevice, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuCtxGetStreamPriorityRange)) {
    std::cout << "<<<<<<<<<<implemented as cuCtxGetStreamPriorityRange"
              << std::endl;
    using param_t =
        FunctionTraits<decltype(cuCtxGetStreamPriorityRange)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    int least, greatest;
    std::get<0>(args) = &least;
    std::get<1>(args) = &greatest;
    CUresult result = std::apply(cuCtxGetStreamPriorityRange, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuCtxPopCurrent)) {
    std::cout << "<<<<<<<<<<implemented as cuCtxPopCurrent" << std::endl;
    using param_t = FunctionTraits<decltype(cuCtxPopCurrent)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUcontext ctx;
    std::get<0>(args) = &ctx;
    CUresult result = std::apply(cuCtxPopCurrent, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuModuleGetLoadingMode)) {
    std::cout << "<<<<<<<<<<implemented as cuModuleGetLoadingMode" << std::endl;
    using param_t =
        FunctionTraits<decltype(cuModuleGetLoadingMode)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUmoduleLoadingMode mode;
    std::get<0>(args) = &mode;
    CUresult result = std::apply(cuModuleGetLoadingMode, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuLibraryGetModule)) {
    std::cout << "<<<<<<<<<<implemented as cuLibraryGetModule" << std::endl;
    using param_t =
        FunctionTraits<decltype(cuLibraryGetModule)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUmodule module;
    std::get<0>(args) = &module;
    CUresult result = std::apply(cuLibraryGetModule, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuEventCreate)) {
    std::cout << "<<<<<<<<<<implemented as cuEventCreate" << std::endl;
    using param_t = FunctionTraits<decltype(cuEventCreate)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUevent event;
    std::get<0>(args) = &event;
    CUresult result = std::apply(cuEventCreate, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuStreamCreate)) {
    std::cout << "<<<<<<<<<<implemented as cuStreamCreate" << std::endl;
    using param_t = FunctionTraits<decltype(cuStreamCreate)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUstream stream;
    std::get<0>(args) = &stream;
    CUresult result = std::apply(cuStreamCreate, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuDeviceGetName)) {
    std::cout << "<<<<<<<<<<implemented as cuDeviceGetName" << std::endl;
    using param_t = FunctionTraits<decltype(cuDeviceGetName)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    char name[std::get<1>(args)];
    std::get<0>(args) = name;
    CUresult result = std::apply(cuDeviceGetName, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuModuleGetGlobal)) {
    std::cout << "<<<<<<<<<<implemented as cuModuleGetGlobal" << std::endl;
    using param_t = FunctionTraits<decltype(cuModuleGetGlobal)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    CUdeviceptr dptr;
    size_t bytes;
    std::get<0>(args) = &dptr;
    std::get<1>(args) = &bytes;
    CUresult result = std::apply(cuModuleGetGlobal, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name,
                       cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)) {
    std::cout << "<<<<<<<<<<implemented as "
                 "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
              << std::endl;
    using param_t = FunctionTraits<
        decltype(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)>::
        ParameterTuple;
    param_t args;
    deserializer_ >> args;
    int numBlocks;
    std::get<0>(args) = &numBlocks;
    CUresult result =
        std::apply(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, args);
    response_.set_curesult(result);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuMemAlloc)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;
    using param_t = FunctionTraits<decltype(cuMemAlloc)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;

    device_ptrs_.push_back(std::make_unique<CUdeviceptr>());
    std::get<0>(args) = device_ptrs_.back().get();
    CUresult result = std::apply(cuMemAlloc, args);
    response_.set_curesult(result);
    response_.set_cuscalar(*device_ptrs_.back());
  } else if (FUNC_COMP(function_name, cuMemcpyHtoD)) {
    // HACK consider controled access pattern
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<decltype(cuMemcpyHtoD)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;

    long long bytesize = std::get<2>(args);
    void *client_ptr = (void *)std::get<1>(args);
    if (is_pinned(client_ptr))
      std::get<1>(args) = convert_pin(client_ptr);
    else
      std::get<1>(args) = vgpu_ptr_;

    CUresult result = std::apply(cuMemcpyHtoD, args);

    float temp = ((float *)vgpu_ptr_)[1];
    std::cout << "cuMemcpyHtoD gives " << temp << std::endl;
    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuModuleLoad)) {
    // HACK consider access control in multi-client case
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<decltype(cuModuleLoad)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    cumodules_.push_back(std::make_unique<CUmodule>());
    CUmodule _module;
    auto full_path = std::string(SOURCE_DIR) + "/vector_add.ptx";
    CUresult result = cuModuleLoad(&_module, full_path.c_str());

    std::cout << (char *)vgpu_ptr_ << std::endl;

    response_.set_cuscalar((uint64_t)_module);
    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuModuleLoadData)) {
    // HACK consider access control in multi-client case
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<decltype(cuModuleLoad)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    cumodules_.push_back(std::make_unique<CUmodule>());
    std::get<0>(args) = cumodules_.back().get();
    std::get<1>(args) = (const char *)vgpu_ptr_;
    CUresult result = std::apply(cuModuleLoadData, args);
    response_.set_cuscalar((uint64_t)*cumodules_.back());
    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuOccupancyMaxPotentialBlockSize)) {
    // HACK consider access control in multi-client case
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<
        decltype(cuOccupancyMaxPotentialBlockSize)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    int minGridSize, blockSize;
    std::get<0>(args) = &minGridSize;
    std::get<1>(args) = &blockSize;
    CUresult result = std::apply(cuOccupancyMaxPotentialBlockSize, args);
    response_ << minGridSize << blockSize;
    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuModuleGetFunction)) {
    // HACK consider access control in multi-client case
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t =
        FunctionTraits<decltype(cuModuleGetFunction)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    cufuncs_.push_back(std::make_unique<CUfunction>());
    std::get<0>(args) = cufuncs_.back().get();
    CUresult result = std::apply(cuModuleGetFunction, args);

    response_.set_cuscalar((uint64_t)*cufuncs_.back());
    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuLaunchKernel)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<decltype(cuLaunchKernel)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    std::vector<uint64_t *> kernel_args;
    deserializer_ >> kernel_args;
    std::get<9>(args) = (void **)kernel_args.data();
    CUresult result = std::apply(cuLaunchKernel, args);

    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuMemcpyDtoH)) {
    // HACK consider controlled access pattern
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<decltype(cuMemcpyDtoH)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;

    void *client_ptr = (void *)std::get<0>(args);
    size_t bytesize = std::get<2>(args);
    if (is_pinned(client_ptr))
      std::get<0>(args) = convert_pin(client_ptr);
    else
      std::get<0>(args) = vgpu_ptr_;

    CUresult result = std::apply(cuMemcpyDtoH, args);
    vgpu_.sync(bytesize);
    float temp = ((float *)vgpu_ptr_)[0];
    std::cout << "cuMemcpyDtoH gives " << temp << std::endl;

    response_.set_curesult(result);
    resolve_cuda_error(result);
    // } else if(FUNC_COMP(function_name, cuCtxDestroy)){
    //     // close(client_fd_);
    //     // client_fd_ = -1;
    //     std::cout << "<<<<<<<<<<implemented as " << function_name <<
    //     std::endl; using param_t =
    //     FunctionTraits<decltype(cuCtxDestroy)>::ParameterTuple; param_t args;
    //     deserializer_ >> args; CUresult result = std::apply(cuCtxDestroy,
    //     args); response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuMemHostAlloc)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<decltype(cuMemHostAlloc)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    if (!client_pin_start_)
      client_pin_start_ = (void *)std::get<0>(args);
    registered_.push_back(
        pinned_memory_.register_pinned_memory(std::get<1>(args)));
    size_t alignment = 4096; // page-aligned

    CUresult result = cuMemHostRegister(registered_.back(), std::get<1>(args),
                                        std::get<2>(args));
    std::cout << "Registering ptr: " << registered_.back()
              << " size: " << std::get<1>(args) << std::endl;

    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::cout << "the error should be " << name << std::endl;
    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuMemcpyHtoDAsync)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t = FunctionTraits<decltype(cuMemcpyHtoDAsync)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    void *client_ptr = (void *)std::get<1>(args);
    std::get<1>(args) = convert_pin(client_ptr);
    CUresult result = std::apply(cuMemcpyHtoDAsync, args);

    response_.set_curesult(result);
  } else if (FUNC_COMP(function_name, cuGetExportTable)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;
    using param_t = FunctionTraits<decltype(cuGetExportTable)>::ParameterTuple;
    CUuuid uuid_;
    deserializer_ >> uuid_;
    const void *ptable;

    CUresult result = cuGetExportTable(&ptable, &uuid_);
    response_.set_curesult(result);
    response_ << ptable;
  } else if (FUNC_COMP(function_name, cudaGetDeviceProperties)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;
    using param_t =
        FunctionTraits<decltype(cudaGetDeviceProperties)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    cudaDeviceProp prop;
    std::get<0>(args) = &prop;
    cudaError_t result = std::apply(cudaGetDeviceProperties, args);

    response_.set_curesult(CUDA_SUCCESS);
    response_ << args;
  } else if (FUNC_COMP(function_name, cuMemHostGetDevicePointer)) {
    std::cout << "<<<<<<<<<<implemented as " << function_name << std::endl;

    using param_t =
        FunctionTraits<decltype(cuMemHostGetDevicePointer)>::ParameterTuple;
    param_t args;
    deserializer_ >> args;
    device_ptrs_.push_back(std::make_unique<CUdeviceptr>());
    std::get<0>(args) = device_ptrs_.back().get();
    void *client_ptr = std::get<1>(args);
    std::get<1>(args) = convert_pin(client_ptr);
    CUresult result = std::apply(cuMemHostGetDevicePointer, args);

    response_.set_curesult(result);
    response_ << args;
    // } else if(FUNC_COMP(function_name, cudaMalloc)){
    //     std::cout << "<<<<<<<<<<implemented as " << function_name <<
    //     std::endl; using param_t = std::tuple<void**, size_t>; param_t args;
    //     deserializer_ >> args;

    //     hdata_ptrs_.push_back(std::make_unique<void*>());
    //     std::get<0>(args) = hdata_ptrs_.back().get();
    //     cudaError_t result = cudaMalloc(hdata_ptrs_.back().get(),
    //     std::get<1>(args)); response_ << result; response_ << args;
    // } else if(FUNC_COMP(function_name, cudaMemcpy)){
    //     std::cout << "<<<<<<<<<<implemented as " << function_name <<
    //     std::endl; using param_t = std::tuple<void**, size_t>; param_t args;
    //     deserializer_ >> args;

    //     hdata_ptrs_.push_back(std::make_unique<void*>());
    //     std::get<0>(args) = hdata_ptrs_.back().get();
    //     cudaError_t result = cudaMalloc(hdata_ptrs_.back().get(),
    //     std::get<1>(args)); response_ << result; response_ << args;
  } else if (FUNC_COMP(function_name, cudaGetDeviceCount)) {
    int ret;
    cudaError_t result = cudaGetDeviceCount(&ret);
    response_ << result;
    response_ << ret;
  } else if (FUNC_COMP(function_name, cudaGetDevice)) {
    int ret;
    cudaError_t result = cudaGetDevice(&ret);
    response_ << result;
    response_ << ret;
  } else if (FUNC_COMP(function_name, cudaGetDeviceProperties)) {
    cudaDeviceProp ret1;
    int ret2;
    deserializer_ >> ret2;
    cudaError_t result = cudaGetDeviceProperties(&ret1, ret2);
    response_ << result;
    response_ << ret1;
  } else {
    if (func_map.count(function_name))
      func_map[function_name]();
  }

  return_result();
  std::cout << "<<<<<<<<<<results returned" << std::endl;
}
//======================================================================================//
