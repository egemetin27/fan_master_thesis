#include <cstdio>
#include <stdio.h>
#include <cstring>
#include <csignal>
#include <unistd.h>

#include <functional>
#include <sys/socket.h>
#include <sys/un.h>

#include <thread>
#include <vector>

#include "gpu_instance.h"

#define SOCKET_PATH "/u/home/mege/workspace/tmp/v.sock_1234"

/**
 * @class CUDAServer
 * @brief responsible for initialize server socket to accept guest
 *        requests and initialize client socket for connection.
 * @details upon a request, a dedicated GPU device, a client socket, 
 *          and a brandnew virtual GPU are assigned to the requesting
 *          guest microvm 
 */
class CUDAServer{
  public:
    CUDAServer(){
        server_fd_ = initialize_server();
    }
    ~CUDAServer(){
        close(server_fd_);
        unlink(SOCKET_PATH);
    };

    void run(){
        while(true){
            //todo mutex needed for multi-thread senario
            int client_fd = initialize_client();
            // client_thread_function(0, client_fd);
            auto client_thread = std::make_unique<std::thread>(client_thread_function, 0, client_fd);
            client_thread->detach();
            threads_.push_back(std::move(client_thread));
        }
    }

  /******************************************************
   *                vsock related setup                 *
   ******************************************************/
  private:
    int server_fd_;
    int initialize_server(){
        int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if(server_fd == -1) perror("server socket failed");

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, SOCKET_PATH, sizeof(server_addr.sun_path) - 1);
        unlink(SOCKET_PATH);

        if(bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1){
            perror("server bind failed");
            close(server_fd);
        }

        if(listen(server_fd, 5) == -1){
            perror("server listen failed");
            close(server_fd);
        }

        std::cout << "Serverlistening on " << SOCKET_PATH << std::endl;
        return server_fd;
    }

    int initialize_client(){
        std::cout << "waiting for client" << std::endl;
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
        if(client_fd == -1){
            perror("accept failed");
            close(server_fd_);
        }

        std::cout << "Client connected!" << std::endl;
        return client_fd;
    }

    static void client_thread_function(int device_id, int client_fd){
        GPUInstance gpu_instance(device_id, client_fd);
        gpu_instance.handle();
    }

    std::vector<std::unique_ptr<std::thread>> threads_;
};

CUDAServer *server;

void handle_signal(int signal){
  std::cout << "\nexiting cuda_server with signal " << signal << "..." << std::endl;
  delete server;

  std::exit(0);
}

int main(){
  std::cout << "cuda_server running ..." << std::endl;
  cuInit(0);
  server = new CUDAServer{};
  signal(SIGINT, handle_signal);
  signal(SIGABRT, handle_signal);
  signal(SIGSEGV, handle_signal);
  server->run();

  return 0;
}