#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>

#include <thread>
#include <vector>

#include "gpu_instance.h"

#ifndef SOCKET_PATH
#define SOCKET_PATH "/u/home/mege/workspace/tmp/v.sock_1234"
#endif
#ifndef DEFAULT_SHARED_MEM
#define DEFAULT_SHARED_MEM "/dev/shm/shared_mem"
#endif
#ifndef DEFAULT_CUDA_PIN
#define DEFAULT_CUDA_PIN "/dev/shm/cuda_pin"
#endif

static std::string g_cli_uds; // --uds=…
static std::string g_cli_shm; // --shm=…
static std::string g_cli_pin; // --pin=…
static std::string g_cli_mig; // --migID=…
const char *uds;

static void parse_cli(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strncmp(a, "--uds=", 6) == 0) {
            g_cli_uds = a + 6;
            continue;
        }
        if (std::strncmp(a, "--shm=", 6) == 0) {
            g_cli_shm = a + 6;
            continue;
        }
        if (std::strncmp(a, "--pin=", 6) == 0) {
            g_cli_pin = a + 6;
            continue;
        }
        if (std::strncmp(a, "--migID=", 8) == 0) {
            g_cli_mig = a + 8;
            continue;
        }
    }
}

/**
 * @class CUDAServer
 * @brief responsible for initialize server socket to accept guest
 *        requests and initialize client socket for connection.
 * @details upon a request, a dedicated GPU device, a client socket,
 *          and a brandnew virtual GPU are assigned to the requesting
 *          guest microvm
 */
class CUDAServer {
  public:
    CUDAServer() { server_fd_ = initialize_server(); }
    ~CUDAServer() {
        close(server_fd_);
        unlink(uds);
    };

    void run() {
        while (true) {
            // todo mutex needed for multi-thread senario
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
    int initialize_server() {
        int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd == -1)
            perror("server socket failed");

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, uds, sizeof(server_addr.sun_path) - 1);
        unlink(uds);

        if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
            perror("server bind failed");
            close(server_fd);
        }

        if (listen(server_fd, 5) == -1) {
            perror("server listen failed");
            close(server_fd);
        }

        std::cout << "Serverlistening on " << uds << std::endl;
        return server_fd;
    }

    int initialize_client() {
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd == -1) {
            perror("accept failed");
            close(server_fd_);
        }

        std::cout << "Client connected!" << std::endl;
        return client_fd;
    }

    static void client_thread_function(int device_id, int client_fd) {
        GPUInstance gpu_instance(device_id, client_fd);
        gpu_instance.handle();
    }

    std::vector<std::unique_ptr<std::thread>> threads_;
};

CUDAServer *server;

void handle_signal(int signal) {
    std::cout << "\nexiting cuda_server with signal " << signal << "..." << std::endl;
    delete server;

    std::exit(0);
}

int main(int argc, char **argv) {
    parse_cli(argc, argv);
    std::string uds_path = g_cli_uds.empty() ? std::string(SOCKET_PATH) : g_cli_uds;
    std::string shm_path = g_cli_shm.empty() ? std::string(DEFAULT_SHARED_MEM) : g_cli_shm;
    std::string pin_path = g_cli_pin.empty() ? std::string(DEFAULT_CUDA_PIN) : g_cli_pin;

    ::setenv("FC_SHARED_MEM", shm_path.c_str(), 1);
    ::setenv("FC_CUDA_PIN", pin_path.c_str(), 1);

    // MIG pinning: prefer CLI --migID, fall back to pre-set env
    if (!g_cli_mig.empty()) {
        const char* cur = std::getenv("CUDA_VISIBLE_DEVICES");
        if (!cur || std::string(cur) != g_cli_mig) {
            ::setenv("CUDA_VISIBLE_DEVICES", g_cli_mig.c_str(), 1);
        }
    }

    std::fprintf(stderr, "cuda_server: uds=%s shm=%s pin=%s MIG=%s\n", uds_path.c_str(), shm_path.c_str(), pin_path.c_str(),
                 std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "<unset>");

    uds = uds_path.c_str();

    std::cout << "cuda_server running ..." << std::endl;
    CUresult init = cuInit(0);
    if (init != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", (int)init);
        return 1;
    }
    server = new CUDAServer{};
    signal(SIGINT, handle_signal);
    signal(SIGABRT, handle_signal);
    signal(SIGSEGV, handle_signal);
    server->run();

    return 0;
}
