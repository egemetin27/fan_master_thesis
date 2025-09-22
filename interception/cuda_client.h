#ifndef CUDA_CLIENT_H
#define CUDA_CLIENT_H

#include <stdio.h>
#include <sys/shm.h>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>

#include "stream.h"
#include "type_decl.h"
#include <cuda.h>

// You likely already had these (adjust if your originals differ)
#define VSOCK_HOST_CID_DEFAULT 2
#define VSOCK_PORT_DEFAULT     1234
#define BUFFER_RECV            200
#define SHM_PATH               "/dev/vdb"

// ---- Runtime-config helpers (no per-allocation recompile needed) ------------
namespace fcgpu_cfg {

inline unsigned parse_port_env(const char* name, unsigned defv){
    const char* e = std::getenv(name);
    if(!e || !*e) return defv;
    char* end=nullptr;
    unsigned long v = std::strtoul(e,&end,10);
    if(end==e || v<1 || v>65535) return defv;
    return static_cast<unsigned>(v);
}

inline unsigned server_port(){
    static unsigned cached = 0;
    if(cached) return cached;

    // priority: FC_SERVER_PORT > FC_SERVER_PORT_FILE > default
    const char* p = std::getenv("FC_SERVER_PORT");
    if(p && *p){
        cached = parse_port_env("FC_SERVER_PORT", VSOCK_PORT_DEFAULT);
        return cached;
    }
    const char* f = std::getenv("FC_SERVER_PORT_FILE");
    if(f && *f){
        std::ifstream in(f);
        unsigned long v=0;
        if(in && (in>>v) && v>0 && v<=65535){
            cached = static_cast<unsigned>(v);
            return cached;
        }
    }
    cached = VSOCK_PORT_DEFAULT;
    return cached;
}

inline unsigned host_cid(){
    return parse_port_env("FC_HOST_CID", VSOCK_HOST_CID_DEFAULT);
}

} // namespace fcgpu_cfg

// -----------------------------------------------------------------------------

class CUDAClient{
  public:
    // If your VsockHandle ctor signature is different, keep your original line here.
    CUDAClient()
      : vsock_handle_(VsockHandle(fcgpu_cfg::host_cid(), fcgpu_cfg::server_port())),
        vgpu_(SHM_PATH, (1 << 20) * 9)
    {}
    ~CUDAClient(){};

    /**
     * @brief redirect the intercepted cuda function to the host.
     * @param func_name the function name of the cuda function.
     * @param args arguments to be passed from various intercepted functions.
     */
    template <typename... Args>
    void CallCudaFunction(const char* func_name, Args... args){
        serializer_ << func_name;
        (serializer_ << ... << args);

        vsock_handle_.transmit(serializer_.data(), serializer_.size());
        serializer_.clean();

        // CUresult result = wait_recv();
    }

    /**
     * @brief synchronize the operation with the host when finished.
     * @return the exact `CUresult` from the host.
     */
    CUresult wait_recv(){
        vsock_handle_.receive(response_.data(), response_.size());
        response_.reset();
        return response_.curesult();
    }

    template <typename... Args>
    CUresult wait_recv(Args... args){
        vsock_handle_.receive(response_.data(), response_.size());
        (response_ >> ... >> args);
        response_.reset();
        return response_.curesult();
    }

    /**
     * @brief send guest data to the virtual gpu for later offloading on the host.
     */
    void to_device(const void* data_ptr, size_t data_size){
        vgpu_.to_device(data_ptr, data_size);
    }

    /**
     * @brief read data from vsock to get data from the virtual gpu.
     */
    void from_device(void* data_ptr, size_t data_size){
        vgpu_.from_device(data_ptr, data_size);
    }

    uint64_t get_scalar_result() { return response_.cuscalar(); };
    CUresult get_curesult() { return response_.curesult(); };
    void close() { vsock_handle_.close_socket(); };
  private:
    VsockHandle vsock_handle_;
    VirtualGPU vgpu_;
    Response response_;
    Serializer serializer_;
};

#endif