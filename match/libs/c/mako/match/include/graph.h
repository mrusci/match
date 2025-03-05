#ifndef __MATCH_${model_name}_GRAPH_RUNTIME_H__
#define __MATCH_${model_name}_GRAPH_RUNTIME_H__

% for include in target.include_list:
#include <${include}.h>
% endfor
#include <tvm/runtime/c_runtime_api.h>

// TVM signature
// void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle
// MATCH signature
// type* inp_A, ..., type* inp_Z, type* out_A, ..., type* out_N
% for node in nodes:
% if node.fallback:
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t ${node.fn_name}(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
% else:
${node.fn_name}(
    % for inp_idx,node_in in enumerate(node.inputs):
    ${"" if inp_idx==0 else ","}${node_in.c_type}* ${node_in.name}_pt
    % endfor
    % for tens_out in node.outputs:
    , ${tens_out.c_type}* ${tens_out.name}_pt
    % endfor
);
% endif
% endfor

int match_${model_name}_run_graph(
    % for rt_i in rt_inputs:
    ${rt_i.c_type}* ${rt_i.name}_pt,
    % endfor
    % for rt_o_idx,rt_o in enumerate(rt_outputs):
    ${"" if rt_o_idx==0 else ","}${rt_o.c_type}* ${rt_o.name}_pt
    % endfor
);
#endif