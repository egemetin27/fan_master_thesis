#ifndef STREAM_H
#define STREAM_H

#include <cstring>
#include <cuda.h>
#include <utility>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <unordered_map>
#include <sys/socket.h>
#include <linux/vm_sockets.h>

#include <stdexcept>
#define BUFFER_SIZE 10000
#define GPU_SIZE 2000000
#define VGPU_FILE "/dev/vdb"

/* Data types defined by CUDA that should be processed as non-ptr. */
template <typename T>
constexpr bool is_cuda_type = std::disjunction_v<
  std::is_same<T, CUcontext>,
  std::is_same<T, CUmodule>,
  std::is_same<T, CUstream>,
  std::is_same<T, CUlibrary>,
  std::is_same<T, CUevent>,
  std::is_same<T, CUfunction>
>;
/**
 * @class VsockHandle
 * @brief vsock socket management for a client with corresponding cid and port.
 * @details responsible for the construction and destruction of the vsock, and handles
 *          all the details for correct send and receive data.
 */
class VsockHandle{
  public:
    VsockHandle(unsigned int cid, unsigned int port) : sock_(initialize_sock(cid, port)){};
    ~VsockHandle(){ if(sock_ != -1) close(sock_); };

    void transmit(const void* data_ptr, size_t data_size){
        char* remain_data = (char*)data_ptr;
        while(data_size > 0){
            ssize_t bytes_send = send(sock_, remain_data, data_size, 0);
            if(bytes_send <= 0)
                throw std::runtime_error("Socket closed or error occurred during recv()");
            data_size -= bytes_send;
            remain_data += bytes_send;
        }
    }

    void receive(void* data_ptr, size_t data_size){
        ssize_t total_size = 0;
        char* data = (char*) data_ptr;
        while(total_size < data_size){
          ssize_t local_size = recv(sock_, data + total_size, data_size - total_size, MSG_WAITALL);
          if (local_size < 0) {
              perror("recv error");
              exit(EXIT_FAILURE);
          }
          if (local_size == 0) {
              // peer closed connection
              fprintf(stderr, "Connection closed prematurely\n");
              exit(EXIT_FAILURE);
          }
          total_size += local_size;
        }
    }

    void close_socket() {
        close(sock_);
        sock_ = -1;
    }

  private:
    int sock_;
    int initialize_sock(unsigned int cid, unsigned int port){
        int sock = socket(AF_VSOCK, SOCK_STREAM, 0);
        if(sock < 0) perror("socket");

        sockaddr_vm sa = {};
        sa.svm_family = AF_VSOCK;
        sa.svm_cid = cid;
        sa.svm_port = port;

        std::cout << "Connecting to host ..." << std::endl;
        std::cout << "Vsock destination: cid=" << sa.svm_cid
            << ", port=" << sa.svm_port << ", family=" << sa.svm_family
            << ", sock=" << sock << std::endl;
        if(connect(sock, (struct sockaddr*)&sa, sizeof(sa)) < 0) perror("connect");

        return sock;
    }
};

/**
 * @class Response
 * @brief Used by the server to write and send response.
 *        Used by the client to receive and read response.
 */
class Response {
  public:
    Response() : mdata_(std::malloc(msize_)),
                 curesult_buffer_(reinterpret_cast<CUresult*>(mdata_)),
                 scalar_result_buffer_(reinterpret_cast<uint64_t*>(curesult_buffer_ + 1)),
                 return_buffer_(reinterpret_cast<void*>(scalar_result_buffer_)){};
    ~Response(){ free(mdata_); };

    CUresult curesult() { return *curesult_buffer_; };
    uint64_t cuscalar() { return *scalar_result_buffer_; };
    void set_curesult(CUresult curesult) { *curesult_buffer_ = curesult; };
    void set_cuscalar(uint64_t cuscalar) { *scalar_result_buffer_ = cuscalar; };

    template <typename T>
    Response& operator>>(T &content){
      if constexpr (is_cuda_type<T>){
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(T));
      } else if constexpr (std::is_same<T, char*>::value){
        //!* is it possible for a char vector to be processed as function input?
        size_t len;
        std::memcpy(&len, return_buffer_, sizeof(size_t));
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(size_t));
        std::memcpy(content, return_buffer_, sizeof(char) * len);
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(char) * len);
      } else if constexpr (std::is_same<T, void*>::value ||
                           std::is_same<T, const char*>::value ||
                           std::is_same<T, const void*>::value) {
      } else if constexpr (std::is_same<T, const void**>::value) {
        *content = (void*)return_buffer_;
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(void*));
      } else if constexpr (std::is_pointer<T>::value){
        std::memcpy(content, return_buffer_, sizeof(std::remove_pointer_t<T>));
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(std::remove_pointer_t<T>));
      } else {
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(T));
      }

      return *this;
    }

    template <typename T>
    Response& operator<<(const T &content){
      if constexpr (is_cuda_type<T>){
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(T));
      } else if constexpr (std::is_same<T, char*>::value){
        size_t len = std::strlen(content) + 1; // add the terminator
        std::memcpy(return_buffer_, &len, sizeof(size_t));
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(size_t));
        std::memcpy(return_buffer_, content, sizeof(char) * len);
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(char) * len);
      } else if constexpr (std::is_same<T, void*>::value ||
                          std::is_same<T, const char*>::value ||
                          std::is_same<T, const void*>::value) {
        //todo check what to do for void type
      } else if constexpr (std::is_pointer<T>::value){
        std::memcpy(return_buffer_, content, sizeof(std::remove_pointer_t<T>));
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(std::remove_pointer_t<T>));
      } else {
        std::memcpy(return_buffer_, &content, sizeof(T));
        return_buffer_ = static_cast<void*>(static_cast<char*>(return_buffer_) + sizeof(T));
      }

      return *this;
    }

    template <typename... Args>
    Response& operator>>(std::tuple<Args...> &args){
        deserialize_tuple_out(args, std::index_sequence_for<Args...>{});
        return *this;
    }

    template <typename... Args>
    Response& operator<<(std::tuple<Args...> &args){
        deserialize_tuple_in(args, std::index_sequence_for<Args...>{});
        return *this;
    }

    void reset() { return_buffer_ = (void*)scalar_result_buffer_; };

    void* data() { return mdata_; };
    static size_t size() { return msize_; };

  private:
    static const size_t msize_ = sizeof(CUresult) + 300 * sizeof(uint64_t);
    void* mdata_;
    CUresult *curesult_buffer_;
    uint64_t *scalar_result_buffer_;
    void *return_buffer_;

    template <typename Tuple, size_t... I>
    void deserialize_tuple_out(Tuple &tuple, std::index_sequence<I...>){
        (operator>>(std::get<I>(tuple)), ...);
    }

    template <typename Tuple, size_t... I>
    void deserialize_tuple_in(Tuple &tuple, std::index_sequence<I...>){
        (operator<<(std::get<I>(tuple)), ...);
    }
};

/**
 * @class Serializer
 * @brief Prepares original cuda function information as a flattened message,
 *        which will then be transferred via vsock to the host.
 */
class Serializer {
  public:
    explicit Serializer() : mdata_((char*)std::malloc(BUFFER_SIZE)), mcount_(0){};
    ~Serializer(){ free(mdata_); };

    /** the generic template for normal types */
    template <typename T>
    Serializer& operator<<(T &content){
        std::memcpy(mdata_ + mcount_, &content, sizeof(T));
        mcount_ += sizeof(T);

        return *this;
    }

    /** for `char*` particularly to get it correct. */
    Serializer& operator<<(const char* str_content){
        size_t size = std::strlen(str_content);
        operator<<(size);

        std::memcpy(mdata_ + mcount_, str_content, size * sizeof(char));
        mcount_ += size * sizeof(char);

        static const char term = '\0';
        std::memcpy(mdata_ + mcount_, &term, sizeof(char));
        mcount_ += sizeof(char);

        return *this;
    }

    /** for `char*` particularly to get it correct. */
    Serializer& operator<<(char* str_content){
        size_t size = std::strlen(str_content);
        operator<<(size);

        std::memcpy(mdata_ + mcount_, str_content, size * sizeof(char));
        mcount_ += size * sizeof(char);

        static const char term = '\0';
        std::memcpy(mdata_ + mcount_, &term, sizeof(char));
        mcount_ += sizeof(char);

        return *this;
    }

    /*********************************************
     *            For composite types            *
     *********************************************/
    template <typename... Args>
    Serializer& operator<<(const std::tuple<Args...> &args){
        serialize_tuple(args, std::index_sequence_for<Args...>{});
        return *this;
    }

    template <typename... Args>
    Serializer& operator<<(Args... args){
        operator<<(args...);
        return *this;
    }

    Serializer& operator<<(std::vector<uint64_t> &kernel_args){
        size_t args_count = kernel_args.size();
        operator<<(args_count);
        std::memcpy(mdata_ + mcount_, kernel_args.data(), args_count * sizeof(uint64_t));
        mcount_ += args_count * sizeof(uint64_t);

        return *this;
    }

    /** returns the size of the buffer that stores the flattened message. */
    size_t size() { return mcount_; };

    /** returns the starting of the buffer that stores the flattened message. */
    void* data() {return mdata_; };

    /** cleans the read count with no deletion. */
    void clean() { mcount_ = 0; };

  private:
    char* mdata_;
    size_t mcount_;

    template <typename Tuple, std::size_t... I>
    void serialize_tuple(Tuple &tuple, std::index_sequence<I...>){
        (operator<<(std::get<I>(tuple)), ...);
    }
};

/**
 * @class Deserializer
 * @brief Deserialize the flattened data in the receive buffer.
 */
class Deserializer {
  public:
    explicit Deserializer() : mdata_((char*)std::malloc(BUFFER_SIZE)), mcount_(0){};
    ~Deserializer(){ free(mdata_); };

    template <typename T>
    Deserializer& operator>>(T &content){
        std::memcpy(&content, mdata_ + mcount_, sizeof(T));
        mcount_ += sizeof(T);

        return *this;
    }

    Deserializer& operator>>(const char* &result){
        size_t size; operator>>(size);

        result = (const char*)(mdata_ + mcount_);
        mcount_ += (size + 1) * sizeof(char); //plus the terminator

        return *this;
    }

    Deserializer& operator>>(char* &result){
        size_t size; operator>>(size);

        result = (char*)(mdata_ + mcount_);
        mcount_ += (size + 1) * sizeof(char); //plus the terminator

        return *this;
    }

    /*********************************************
     *            For composite types            *
     *********************************************/
    template <typename... Args>
    Deserializer& operator>>(std::tuple<Args...> &args){
        deserialize_tuple(args, std::index_sequence_for<Args...>{});
        return *this;
    }

    //! read through the doc to check if any other cuda func use this type of data.
    //! if so, have another func for getting the kernel!
    Deserializer& operator>>(std::vector<uint64_t*> &kernel_args){
        size_t size; operator>>(size);

        kernel_args.resize(size);
        for(int i = 0; i < size; i++){
            kernel_args[i] = reinterpret_cast<uint64_t*>(mdata_ + mcount_);
            mcount_ += sizeof(uint64_t);
        }

        return *this;
    }

    void clean() { mcount_ = 0; };
    void* data() {return mdata_; };
    size_t size() { return BUFFER_SIZE; };
  private:
    char* mdata_;
    size_t mcount_;

    template <typename Tuple, size_t... I>
    void deserialize_tuple(Tuple &tuple, std::index_sequence<I...>){
        (operator>>(std::get<I>(tuple)), ...);
    }
};

class PinnedMemory {
  public:
    PinnedMemory(const char* shm_path, size_t mem_size)
        : alloc_pos_(0), pin_size_(mem_size){
        fd_ = open(shm_path, O_RDWR | O_SYNC | O_DIRECT);
        if (fd_ < 0) std::cerr << "Failed to open shared memory file\n";
        pin_start_ = mmap(NULL, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_SYNC, fd_, 0);
        if (pin_start_ == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }
    }
    ~PinnedMemory(){
        if (munmap(pin_start_, pin_size_) == -1) {
          perror("munmap");
        }
        close(fd_);
    }

    void* register_pinned_memory(size_t byte_size){
        void* ret = (char*)pin_start_ + alloc_pos_;
        alloc_pos_ += byte_size;
        hashmap[ret] = byte_size;
        return ret;
    }

    void sync(const void* ptr) { msync((void*)ptr, hashmap[(void*)ptr], MS_SYNC); }
    void* get() { return pin_start_; }
    void remap() {
        if (munmap(pin_start_, pin_size_) == -1) perror("munmap");
        close(fd_);
        fd_ = open("/dev/vdd", O_RDWR | O_SYNC | O_DIRECT);
        if (fd_ < 0) std::cerr << "Failed to open shared memory file\n";
        pin_start_ = mmap(pin_start_, pin_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_SYNC | MAP_FIXED, fd_, 0);
        if (pin_start_ == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }
    }

    bool is_pinned(void* ptr) { return ptr >= pin_start_ && (char*)ptr <= (char*)pin_start_ + pin_size_; }
    bool is_pinned(const void* ptr) { return ptr >= pin_start_ && (char*)ptr <= (char*)pin_start_ + pin_size_; }

  private:
    size_t alloc_pos_;

    int fd_;
    void* pin_start_;
    size_t pin_size_;
    std::string shm_path;
    std::unordered_map<void*, size_t> hashmap;
};

// deprecated due to unknown cache coherency issues
class VirtualGPU {
  public:
    VirtualGPU(const char* shm_path, size_t mem_size)
        : vgpu_size_(mem_size) { initialize_vgpu(shm_path); };
    ~VirtualGPU(){
        close_vgpu();
    }

    void to_device(const void* data_ptr, size_t byte_size) {
      if (vgpu_ptr_ == nullptr) {
          std::cout << "[ERROR] vgpu_ptr_ is null!\n";
          std::abort();
      }
      if(byte_size > vgpu_size_) {
          std::cout << "[Error] vgpu is not large enough\n";
          std::abort();
      }
      if (data_ptr == nullptr) {
          std::cout << "[ERROR] data_ptr is null!\n";
          std::abort();
      }
      memcpy(vgpu_ptr_, data_ptr, byte_size);
      sync(byte_size);
    }
    void from_device(void* data_ptr, size_t byte_size) {
      std::cout << "vgpu_ptr_ = " << vgpu_ptr_ << std::endl;
      remap();
      memcpy(data_ptr, vgpu_ptr_, byte_size);
      std::cout << "vgpu_ptr_ = " << vgpu_ptr_ << std::endl;
    }

    void* get() { return vgpu_ptr_; };
    void sync(size_t byte_size){ msync(vgpu_ptr_, byte_size, MS_SYNC); };
  private:
    void initialize_vgpu(const char* shm_path){
        fd_ = open(shm_path, O_RDWR | O_SYNC | O_DIRECT);
        if (fd_ < 0) std::cerr << "Failed to open shared memory file\n";
        vgpu_ptr_ = mmap(NULL, vgpu_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_SYNC, fd_, 0);
        if (vgpu_ptr_ == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }
    }

    void close_vgpu(){
      if (munmap(vgpu_ptr_, vgpu_size_) == -1) {
        perror("munmap");
      }
      close(fd_);
    };
    
    void remap(){
      if(munmap(vgpu_ptr_, vgpu_size_) == -1) perror("munmap");
      close(fd_);
      fd_ = open(VGPU_FILE, O_RDWR | O_SYNC | O_DIRECT);
      if (fd_ < 0) std::cerr << "Failed to open shared memory file\n";
      vgpu_ptr_ = mmap(vgpu_ptr_, vgpu_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_SYNC, fd_, 0);
      if (vgpu_ptr_ == MAP_FAILED) {
          perror("mmap");
          exit(1);
      }
    }

    int fd_;
    void* vgpu_ptr_;
    size_t vgpu_size_;
    std::string shm_path_;
};

namespace vsock
{
/**
 * @class VirtualGPU
 * @brief use `vsock` as the underlying implementation.
 * @details the commands and data transfer share the same socket, but
 *          different buffers. May be no buffer at all. Because the array
 *          of data can be a buffer itself.
 */
class VirtualGPU {
  public:
    VirtualGPU() : vgpu_(nullptr), vgpu_size_(0){};
    ~VirtualGPU(){ if(!vgpu_) free(vgpu_); };

    void to_device(void *data_ptr, size_t data_size){
        if(data_size > vgpu_size_) allocate_vgpu(data_size);
        memcpy(vgpu_, data_ptr, data_size);
    }
    void from_device(size_t data_size){
        if(data_size > vgpu_size_) allocate_vgpu(data_size);
    }

    void* prepare(size_t data_size){
        if(data_size > vgpu_size_) allocate_vgpu(data_size);
        return vgpu_;
    }

    void* get() { return vgpu_; };

  private:
    void* vgpu_;
    size_t vgpu_size_;

    void allocate_vgpu(size_t size){
        if(vgpu_) free(vgpu_);
        vgpu_ = malloc(size);
        vgpu_size_ = size;
    }
};
}
#endif