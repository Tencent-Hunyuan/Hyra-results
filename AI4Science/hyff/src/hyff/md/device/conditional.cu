#include <cuda_runtime.h>
#include <cstdint>
#include <new>

// Ledger: requested, committed, attempted forces, loop iterations, stop reason,
// then a copy of the six CheckedPhases flags.
struct Loop {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    int *ledger = nullptr;
    int ledger_columns = 11;
    ~Loop() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

__global__ void initialize(int *ledger, int count, int columns) {
    for (int i = 0; i < columns; ++i) ledger[i] = 0;
    ledger[0] = count;
}

__global__ void decide_force(int *flags, int *ledger,
                             cudaGraphConditionalHandle handle) {
    ++ledger[3];
    bool valid = flags[0] == 2 && flags[1] == 2 && flags[3] == 2;
    cudaGraphSetConditional(handle, valid ? 1 : 0);
}

__global__ void commit(int *flags, int *ledger) {
    ++ledger[2];
    if (flags[2] == 2 && flags[4] == 2 && flags[5] == 2) ++ledger[1];
}

__global__ void decide_rebuild(int *flags, cudaGraphConditionalHandle handle) {
    bool needed = flags[0] == 1 && flags[1] == 2 && flags[3] == 2;
    cudaGraphSetConditional(handle, needed ? 1 : 0);
}

__global__ void finish_rebuild(int *flags, int *ledger, int *status) {
    ++ledger[11];
    for (int i = 0; i < 4; ++i) ledger[12 + i] = status[i];
    if (status[0] == 0) flags[0] = 2;
}

__global__ void decide_loop(int *flags, int *ledger, int *neighbor_status,
                            cudaGraphConditionalHandle handle) {
    int reason = 0;
    if (flags[1] != 2 || flags[3] != 2) reason = 2;
    else if (neighbor_status && neighbor_status[0] != 0) reason = 4;
    else if (flags[0] != 2) reason = 1;
    else if (flags[2] != 2 || flags[4] != 2 || flags[5] != 2) reason = 3;
    ledger[4] = reason;
    for (int i = 0; i < 6; ++i) ledger[5 + i] = flags[i];
    cudaGraphSetConditional(handle, reason == 0 && ledger[1] < ledger[0]);
}

static cudaError_t add_kernel(cudaGraphNode_t *node, cudaGraph_t graph,
                             cudaGraphNode_t *dependency, void *function,
                             void **args) {
    cudaKernelNodeParams p{};
    p.func = function;
    p.gridDim = dim3(1);
    p.blockDim = dim3(1);
    p.kernelParams = args;
    return cudaGraphAddKernelNode(node, graph, dependency, dependency ? 1 : 0, &p);
}

extern "C" const char *hyff_cuda_error(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int hyff_capture_graph(void *stream, void **graph) {
    cudaStreamCaptureStatus status;
    return cudaStreamGetCaptureInfo_v2(static_cast<cudaStream_t>(stream), &status,
        nullptr, reinterpret_cast<cudaGraph_t *>(graph), nullptr, nullptr);
}

extern "C" int hyff_clone_graph(void *source, void **clone) {
    return cudaGraphClone(reinterpret_cast<cudaGraph_t *>(clone),
                          static_cast<cudaGraph_t>(source));
}

extern "C" int hyff_destroy_graph(void *graph) {
    return cudaGraphDestroy(static_cast<cudaGraph_t>(graph));
}

static int make_loop(void *pre, void *rebuild, void *force_post, int *flags,
                     int *ledger, int *neighbor_status, void **result) {
    *result = nullptr;
    Loop *loop = new (std::nothrow) Loop;
    if (!loop) return cudaErrorMemoryAllocation;
    loop->ledger = ledger;
    if (rebuild) loop->ledger_columns = 16;
    cudaError_t error = cudaSuccess;
#define CHECK(call) do { error = (call); if (error != cudaSuccess) { delete loop; return error; } } while (0)
    CHECK(cudaGraphCreate(&loop->graph, 0));
    cudaGraphConditionalHandle while_handle, if_handle;
    CHECK(cudaGraphConditionalHandleCreate(&while_handle, loop->graph, 1, cudaGraphCondAssignDefault));
    CHECK(cudaGraphConditionalHandleCreate(&if_handle, loop->graph, 0, cudaGraphCondAssignDefault));
    cudaGraphNodeParams while_params{};
    while_params.type = cudaGraphNodeTypeConditional;
    while_params.conditional.handle = while_handle;
    while_params.conditional.type = cudaGraphCondTypeWhile;
    while_params.conditional.size = 1;
    cudaGraphNode_t while_node;
    CHECK(cudaGraphAddNode(&while_node, loop->graph, nullptr, 0, &while_params));
    cudaGraph_t body = while_params.conditional.phGraph_out[0];
    cudaGraphNode_t pre_node, decision_node, if_node, commit_node, post_node, repeat_node;
    CHECK(cudaGraphAddChildGraphNode(&pre_node, body, nullptr, 0, static_cast<cudaGraph_t>(pre)));
    cudaGraphNode_t force_dependency = pre_node;
    if (rebuild) {
        cudaGraphConditionalHandle rebuild_handle;
        CHECK(cudaGraphConditionalHandleCreate(&rebuild_handle, loop->graph, 0, cudaGraphCondAssignDefault));
        cudaGraphNode_t decide_node, rebuild_node, child_node, publish_node;
        void *decide_args[] = {&flags, &rebuild_handle};
        CHECK(add_kernel(&decide_node, body, &pre_node, reinterpret_cast<void *>(decide_rebuild), decide_args));
        cudaGraphNodeParams rebuild_params{};
        rebuild_params.type = cudaGraphNodeTypeConditional;
        rebuild_params.conditional.handle = rebuild_handle;
        rebuild_params.conditional.type = cudaGraphCondTypeIf;
        rebuild_params.conditional.size = 1;
        CHECK(cudaGraphAddNode(&rebuild_node, body, &decide_node, 1, &rebuild_params));
        cudaGraph_t rebuild_body = rebuild_params.conditional.phGraph_out[0];
        CHECK(cudaGraphAddChildGraphNode(&child_node, rebuild_body, nullptr, 0, static_cast<cudaGraph_t>(rebuild)));
        void *publish_args[] = {&flags, &ledger, &neighbor_status};
        CHECK(add_kernel(&publish_node, rebuild_body, &child_node, reinterpret_cast<void *>(finish_rebuild), publish_args));
        force_dependency = rebuild_node;
    }
    void *decision_args[] = {&flags, &ledger, &if_handle};
    CHECK(add_kernel(&decision_node, body, &force_dependency, reinterpret_cast<void *>(decide_force), decision_args));
    cudaGraphNodeParams if_params{};
    if_params.type = cudaGraphNodeTypeConditional;
    if_params.conditional.handle = if_handle;
    if_params.conditional.type = cudaGraphCondTypeIf;
    if_params.conditional.size = 1;
    CHECK(cudaGraphAddNode(&if_node, body, &decision_node, 1, &if_params));
    cudaGraph_t force_body = if_params.conditional.phGraph_out[0];
    CHECK(cudaGraphAddChildGraphNode(&post_node, force_body, nullptr, 0,
                                    static_cast<cudaGraph_t>(force_post)));
    void *commit_args[] = {&flags, &ledger};
    CHECK(add_kernel(&commit_node, force_body, &post_node, reinterpret_cast<void *>(commit), commit_args));
    void *repeat_args[] = {&flags, &ledger, &neighbor_status, &while_handle};
    CHECK(add_kernel(&repeat_node, body, &if_node, reinterpret_cast<void *>(decide_loop), repeat_args));
    CHECK(cudaGraphInstantiate(&loop->executable, loop->graph, 0));
    *result = loop;
    return cudaSuccess;
#undef CHECK
}

extern "C" int hyff_make_loop(void *pre, void *force_post, int *flags,
                              int *ledger, void **result) {
    return make_loop(pre, nullptr, force_post, flags, ledger, nullptr, result);
}

extern "C" int hyff_make_loop_with_rebuild(void *pre, void *rebuild, void *force_post,
                                           int *flags, int *ledger, int *status,
                                           void **result) {
    if (!rebuild || !status) return cudaErrorInvalidValue;
    return make_loop(pre, rebuild, force_post, flags, ledger, status, result);
}

extern "C" int hyff_launch_loop(void *instance, void *stream, int count) {
    if (count < 1) return cudaErrorInvalidValue;
    Loop *loop = static_cast<Loop *>(instance);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    initialize<<<1, 1, 0, s>>>(loop->ledger, count, loop->ledger_columns);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    return cudaGraphLaunch(loop->executable, s);
}

extern "C" int hyff_destroy_loop(void *instance) {
    delete static_cast<Loop *>(instance);
    return cudaSuccess;
}
