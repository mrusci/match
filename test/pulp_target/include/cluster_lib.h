#ifndef __PULP_CLUSTER_LIB_H__
#define __PULP_CLUSTER_LIB_H__

#include <match/ctx.h>
#include <pulp_target/pulp_target.h>
#include <pulp_target/dory_dma.h>
#include <pulp_target/gap9_cluster.h>

#define CLUSTER_LIB_DEBUG

void offload_to_pulp_cluster(void (inner_function)(unsigned int* args_inner_function),unsigned int* args);

void cluster_lib_init(MatchCtx* ctx);

void* init_l1_scratchpad_memory(MatchCtx* ctx);

void handle_dma_transfer(
    MatchCtx* ctx, MatchTensor* tensor,
    void* tensor_l2_pt, void* tensor_l1_pt,
    int match_transfer_type, int match_tensor_type,
    int ext_mem, int int_mem 
);

void wait_l1_dma_transfers(MatchCtx* ctx);

void free_l1_scrachpad_memory(MatchCtx* ctx, void* l1_memory_pt);

void wait_pulp_nn_computation(MatchCtx* ctx);

void pulp_nn_dense_wrapper(void* args);

void pulp_nn_dense_out_int_wrapper(void* args);

void pulp_nn_dw_conv2d_less_4_wrapper(void* args);

void pulp_nn_dw_conv2d_wrapper(void* args);

void pulp_nn_pw_conv2d_wrapper(void* args);

void pulp_nn_hoparallel_conv2d_wrapper(void* args);

void pulp_nn_add_wrapper(void* args);

void pulp_nn_wrapper(MatchCtx* ctx);

#endif