/*
 * Mohamed Amine Hamdi <mohamed.hamdi@polito.it>
 *
 * Copyright (C) 2024 Politecnico Di Torino
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
*/
// include params file
#include <nodes/${model_name}/${name}_params.h>

% for block_idx,block in enumerate(schedule.blocks):
<% brackets_cnt = 0 %>

void block_${block_idx}_compute(match_context_t* ctx){
    % for loop_idx,lp in enumerate(block.loops[block.loop_idx_end_sw_controlled_loads:]):
    % if not exec_module.backend_constraints_check(match_node,schedule,block,lp):
    <% continue %>
    % else:
    <% brackets_cnt += 1 %>
    % endif
    ${c_ident(loop_idx)}for(loop_${lp.name}_set(0);
        ${c_ident(loop_idx)}loop_${lp.name}_iter<${lp.size};
        ${c_ident(loop_idx)}loop_${lp.name}_set(++loop_${lp.name}_iter)){
        % for instr in lp.init_instrs:
        ${c_ident(loop_idx)}${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr};
        % endfor
    % endfor
    ## close braces and save output
    % for loop_idx_ in range(loop_idx,-1,-1):
        % for instr in schedule.blocks[block_idx].loops[loop_idx_+1].instrs:
        ${c_ident(brackets_cnt-1)}${instr.lhs_expr.c_expr} ${instr.eq_expr.c_expr} ${instr.rhs_expr.c_expr}; // loop ${loop_idx_+1} ${block.loops[loop_idx_+1].name}
        % endfor
    ${c_ident(brackets_cnt-1)}}
    <% brackets_cnt -= 1 %>
    % if brackets_cnt<0:
    <% break %>
    % endif
    % endfor
}
% endfor


int __attribute__ ((noinline)) ${node_fullname}(
    % for var in match_node.var_tensors.values():
    void* var_${var.name}_pt,
    % endfor
    % for idx,out in enumerate(match_node.output_tensors.values()):
    ${", " if idx>0 else ""}void* out_${out.name}_pt
    % endfor
)
{
    % for var in match_node.var_tensors.values():
    ${name}_${var.name}_var->base_pt = var_${var.name}_pt;
    int mem_level_${var.name} = ${top_level_memory_vars};
    ${name}_${var.name}_var->pts[mem_level_${var.name}] = var_${var.name}_pt;
    % endfor
    % for out in match_node.output_tensors.values():
    ${name}_${out.name}_out->base_pt = out_${out.name}_pt;
    int mem_level_${out.name} = ${top_level_memory_out};
    ${name}_${out.name}_out->pts[mem_level_${out.name}] = out_${out.name}_pt;
    % endfor
    
    % if async_computation:
    int async_computation_idx = 0;
    % endif

    % for mem_level in mem_levels:
    % if mem_level.sw_controlled:
    ${mem_apis.init_mem_levels[mem_level.name]}(&ctx);
    % endif
    % endfor

    % for block_idx,block in enumerate(schedule.blocks):
    // block ${block_idx}

    % for loop_idx,lp in enumerate(block.loops):
    % for mem_transfer in lp.mem_transfers:
    % if mem_transfer.tensor.tensor_type != "output":
    ${c_ident(loop_idx)}match_get_pt_for_tile(&${mem_transfer.tensor.name}->pts,${mem_transfer.top_mem});
    % endif
    % if mem_transfer.tensor.tensor_type == "var":
    ${c_ident(loop_idx)}${mem_apis.var_mem_transfer}(${mem_transfer.top_mem},${mem_transfer.mem});
    % elif mem_transfer.tensor.tensor_type == "const":
    ${c_ident(loop_idx)}${mem_apis.const_mem_transfer}(${mem_transfer.top_mem},${mem_transfer.mem});
    % endif
    % endfor
    ## finished sw controlled loads
    % if exec_module.backend_constraints_check(match_node,schedule,block,lp) and block.loop_idx_end_sw_controlled_loads>=loop_idx:
    <% break %>
    % endif 
    ${c_ident(loop_idx)}for(loop_${lp.name}_set(0);
        ${c_ident(loop_idx)}loop_${lp.name}_iter<${lp.size};
        ${c_ident(loop_idx)}loop_${lp.name}_set(++loop_${lp.name}_iter)){
    % endfor
        
    % if async_computation:
    ${c_ident(loop_idx)}if(async_computation_idx){
        ${c_ident(loop_idx)}${sync_apis.wait_prev_tile_computation}(&ctx);
        % if parallel_execution:
        ${c_ident(loop_idx)}${sync_apis.wait_parallel}(&ctx);
        % endif
    ${c_ident(loop_idx)}}
    % endif

    % if exec_module.backend == "MATCH":
    % if parallel_execution:
    ${c_ident(loop_idx)}${platform_apis.parallelize_task}(block_${block_idx}_compute,&ctx);
    % else:
    ${c_ident(loop_idx)}block_${block_idx}_compute(&ctx);
    % endif
    % else:
    ${c_ident(loop_idx)}${comp_apis.compute_tile}(&ctx);
    % endif

    % if async_computation:
    ${c_ident(loop_idx)}if(async_computation_idx)  ${sync_apis.wait_store_prev_tile}(&ctx);
    % else:
    ${c_ident(loop_idx)}${sync_apis.wait_tile_computation}(&ctx);
    ${c_ident(loop_idx)}${mem_apis.store_tile}(&ctx);
    % endif
        
    ## close braces and save output
    % for loop_idx_ in range(loop_idx-1,-1,-1):
    % if exec_module.backend_constraints_check(match_node,schedule,block,block.loops[loop_idx_]) and block.loop_idx_end_sw_controlled_loads>=loop_idx_:
    <% break %>
    % endif 
    ${c_ident(loop_idx_)}}
    % endfor

    % endfor

    % for mem_level in mem_levels:
    % if mem_level.sw_controlled:
    ${mem_apis.free_mem_levels[mem_level.name]}(&ctx);
    % endif
    % endfor

    return 0;
}