#ifndef __MATCH_CTX_H__
#define __MATCH_CTX_H__
#include <match/utils.h>

typedef enum{
    MATCH_VAR_TENSOR,
    MATCH_CONST_TENSOR,
    MATCH_OUT_TENSOR,
}MATCH_TENSOR_TYPE;

typedef enum{
    MATCH_SW_LOAD_TENSOR,
    MATCH_SW_STORE_TENSOR,
}MATCH_MEM_OPS_TYPE;

typedef enum{
    MATCH_OP_CONV2D,
    MATCH_OP_BIAS_ADD,
    MATCH_OP_ADD,
    MATCH_OP_MULTIPLY,
    MATCH_OP_RELU,
}MATCH_OPS_CODE;

typedef struct{
    int size;
    int dynamic;
    int curr_size;
    int global_idx;
}MatchDim;

typedef struct MatchDims_t{
    int num_dims;
    char** dims_names;
    MatchDim (*get_dim)(struct MatchDims_t *,const char*);
    int (*get_dim_idx)(struct MatchDims_t *);
    MatchDim* dims;
}MatchDims;

typedef struct{
    MatchDim* dim;
    int size;
    int start_idx;
}MatchTensorTile;

typedef struct{
    MatchTensorTile** tiles;
    void* base_pt;
    void** pts;
    int num_tiles;
    int curr_tile;
    int num_dims;
    int bits;
}MatchTensor;

typedef struct MatchTensors_t{
    int num_tensors;
    char** tensors_names;
    MatchTensor* (*get_tensor)(struct MatchTensors_t *,const char*);
    int (*get_tensor_idx)(struct MatchTensors_t *);
    MatchTensor* tensors;
}MatchTensors;

typedef struct{
    int idx;
    int strides[2];
    int dilation[2];
    int padding[4];
    int kernel_size[2];
    int depthwise;
    int groups;
    const char* data_layout;
    const char* kernel_layout;
}MatchConv2DAttrs;

typedef struct{
    int idx;
    int inp_features;
    int out_features;
}MatchDenseAttrs;

typedef struct{
    int idx;
    int right_shift;
}MatchRightShiftAttrs;

typedef struct{
    int idx;
    int clip_min;
    int clip_max;
}MatchClipAttrs;

typedef struct{
    int idx;
}MatchCastAttrs;

typedef struct{
    int idx;
    int axis;
}MatchBiasAddAttrs;

typedef struct{
    int idx;
    int axis;
}MatchMultiplyAttrs;

typedef struct{
    int idx;
    int axis;
}MatchAddAttrs;

typedef struct{
    int idx;
}MatchReLUAttrs;

typedef struct{
    int op_code;
    // MatchConv2DAttrs or others...
    void* attrs;
}MatchOp;

typedef struct MatchOps_t{
    int num_ops;
    char** ops_names;
    MatchOp* (*get_op)(struct MatchOps_t *,const char*);
    int (*get_op_idx)(struct MatchOps_t *);
    MatchOp* ops;
}MatchOps;

typedef struct{
    void* ctx_extension;

    MatchTensors* tensors;
    
    MatchOps* ops;

    MatchDims* dims;

    int pattern_family;
    int pattern_name;
}MatchCtx;

MatchTensor* default_match_ctx_get_tensor(struct MatchTensors_t *self,const char *name);

int default_match_ctx_get_tensor_idx(struct MatchTensors_t *self,const char *name);

MatchOp* default_match_ctx_get_op(struct MatchOps_t *self,const char *name);

int default_match_ctx_get_op_idx(struct MatchOps_t *self,const char *name);

MatchDim* default_match_ctx_get_dim(struct MatchDims_t *self,const char *name);

int default_match_ctx_get_dim_idx(struct MatchDims_t *self,const char *name);

#endif