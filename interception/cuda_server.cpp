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
#include <iostream>

#include "gpu_instance.h"

// Debug logging macros
#define DEBUG_LOG(msg, ...) do { \
    if (g_debug_mode) { \
        fprintf(stderr, "[DEBUG] " msg "\n", ##__VA_ARGS__); \
        fflush(stderr); \
    } \
} while(0)

#define DEBUG_ENTER(func_name) DEBUG_LOG("Entering function: %s", func_name)
#define DEBUG_EXIT(func_name) DEBUG_LOG("Exiting function: %s", func_name)
#define DEBUG_ERROR(msg, ...) do { \
    if (g_debug_mode) { \
        fprintf(stderr, "[DEBUG ERROR] " msg "\n", ##__VA_ARGS__); \
        fflush(stderr); \
    } \
} while(0)

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
static bool g_debug_mode = false; // --debug
const char *uds;

static void parse_cli(int argc, char **argv) {
    DEBUG_ENTER("parse_cli");
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (std::strncmp(a, "--uds=", 6) == 0) {
            g_cli_uds = a + 6;
            DEBUG_LOG("Found --uds=%s", g_cli_uds.c_str());
            continue;
        }
        if (std::strncmp(a, "--shm=", 6) == 0) {
            g_cli_shm = a + 6;
            DEBUG_LOG("Found --shm=%s", g_cli_shm.c_str());
            continue;
        }
        if (std::strncmp(a, "--pin=", 6) == 0) {
            g_cli_pin = a + 6;
            DEBUG_LOG("Found --pin=%s", g_cli_pin.c_str());
            continue;
        }
        if (std::strncmp(a, "--migID=", 8) == 0) {
            g_cli_mig = a + 8;
            DEBUG_LOG("Found --migID=%s", g_cli_mig.c_str());
            continue;
        }
        if (std::strcmp(a, "--debug") == 0) {
            g_debug_mode = true;
            DEBUG_LOG("Debug mode enabled");
            continue;
        }
    }
    DEBUG_EXIT("parse_cli");
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
    CUDAServer() { 
        DEBUG_ENTER("CUDAServer::CUDAServer");
        server_fd_ = initialize_server(); 
        DEBUG_EXIT("CUDAServer::CUDAServer");
    }
    ~CUDAServer() {
        DEBUG_ENTER("CUDAServer::~CUDAServer");
        close(server_fd_);
        unlink(uds);
        DEBUG_EXIT("CUDAServer::~CUDAServer");
    };

    void run() {
        DEBUG_ENTER("CUDAServer::run");
        while (true) {
            DEBUG_LOG("Waiting for client connection...");
            // todo mutex needed for multi-thread senario
            int client_fd = initialize_client();
            DEBUG_LOG("Client connected, creating thread for client_fd=%d", client_fd);
            // client_thread_function(0, client_fd);
            auto client_thread = std::make_unique<std::thread>(client_thread_function, 0, client_fd);
            client_thread->detach();
            threads_.push_back(std::move(client_thread));
            DEBUG_LOG("Client thread created and detached");
        }
    }

    /******************************************************
     *                vsock related setup                 *
     ******************************************************/
  private:
    int server_fd_;
    int initialize_server() {
        DEBUG_ENTER("CUDAServer::initialize_server");
        int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd == -1) {
            DEBUG_ERROR("server socket failed");
            perror("server socket failed");
        }

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, uds, sizeof(server_addr.sun_path) - 1);
        unlink(uds);

        if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
            DEBUG_ERROR("server bind failed");
            perror("server bind failed");
            close(server_fd);
        }

        if (listen(server_fd, 5) == -1) {
            DEBUG_ERROR("server listen failed");
            perror("server listen failed");
            close(server_fd);
        }

        std::cout << "Serverlistening on " << uds << std::endl;
        DEBUG_LOG("Server successfully listening on %s with fd=%d", uds, server_fd);
        DEBUG_EXIT("CUDAServer::initialize_server");
        return server_fd;
    }

    int initialize_client() {
        DEBUG_ENTER("CUDAServer::initialize_client");
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd == -1) {
            DEBUG_ERROR("accept failed");
            perror("accept failed");
            close(server_fd_);
        }

        std::cout << "Client connected!" << std::endl;
        DEBUG_LOG("Client accepted with fd=%d", client_fd);
        DEBUG_EXIT("CUDAServer::initialize_client");
        return client_fd;
    }

    static void client_thread_function(int device_id, int client_fd) {
        DEBUG_ENTER("CUDAServer::client_thread_function");
        DEBUG_LOG("Starting GPU instance with device_id=%d, client_fd=%d", device_id, client_fd);
        GPUInstance gpu_instance(device_id, client_fd);
        gpu_instance.handle();
        DEBUG_LOG("GPU instance handling completed");
        DEBUG_EXIT("CUDAServer::client_thread_function");
    }

    std::vector<std::unique_ptr<std::thread>> threads_;
};

CUDAServer *server;

void handle_signal(int signal) {
    DEBUG_ENTER("handle_signal");
    std::cout << "\nexiting cuda_server with signal " << signal << "..." << std::endl;
    DEBUG_LOG("Received signal %d, shutting down server", signal);
    delete server;
    DEBUG_LOG("Server deleted, exiting");
    DEBUG_EXIT("handle_signal");

    std::exit(0);
}

int main(int argc, char **argv) {
    parse_cli(argc, argv);
    DEBUG_ENTER("main");
    
    std::string uds_path = g_cli_uds.empty() ? std::string(SOCKET_PATH) : g_cli_uds;
    std::string shm_path = g_cli_shm.empty() ? std::string(DEFAULT_SHARED_MEM) : g_cli_shm;
    std::string pin_path = g_cli_pin.empty() ? std::string(DEFAULT_CUDA_PIN) : g_cli_pin;

    DEBUG_LOG("Configuration - uds_path: %s, shm_path: %s, pin_path: %s", 
              uds_path.c_str(), shm_path.c_str(), pin_path.c_str());

    ::setenv("FC_SHARED_MEM", shm_path.c_str(), 1);
    ::setenv("FC_CUDA_PIN", pin_path.c_str(), 1);
    DEBUG_LOG("Environment variables set");

    // MIG pinning: prefer CLI --migID, fall back to pre-set env
    if (!g_cli_mig.empty()) {
        const char* cur = std::getenv("CUDA_VISIBLE_DEVICES");
        if (!cur || std::string(cur) != g_cli_mig) {
            ::setenv("CUDA_VISIBLE_DEVICES", g_cli_mig.c_str(), 1);
        }
        DEBUG_LOG("MIG device set to: %s", g_cli_mig.c_str());
    }

    std::fprintf(stderr, "cuda_server: uds=%s shm=%s pin=%s MIG=%s\n", uds_path.c_str(), shm_path.c_str(), pin_path.c_str(),
                 std::getenv("CUDA_VISIBLE_DEVICES") ? std::getenv("CUDA_VISIBLE_DEVICES") : "<unset>");

    uds = uds_path.c_str();

    std::cout << "cuda_server running ..." << std::endl;
    DEBUG_LOG("Starting CUDA initialization");
    
    // Handle CUDA initialization with proper error checking
    #ifdef CUDA_SUCCESS
    CUresult init = cuInit(0);
    if (init != CUDA_SUCCESS) {
        DEBUG_ERROR("cuInit failed: %d", (int)init);
        fprintf(stderr, "cuInit failed: %d\n", (int)init);
        return 1;
    }
    DEBUG_LOG("CUDA initialization successful");
    #else
    DEBUG_LOG("CUDA headers not available, skipping cuInit");
    #endif
    
    DEBUG_LOG("Creating CUDAServer instance");
    server = new CUDAServer{};
    
    DEBUG_LOG("Setting up signal handlers");
    signal(SIGINT, handle_signal);
    signal(SIGABRT, handle_signal);
    signal(SIGSEGV, handle_signal);
    
    DEBUG_LOG("Starting server run loop");
    server->run();

    DEBUG_EXIT("main");
    return 0;
}
