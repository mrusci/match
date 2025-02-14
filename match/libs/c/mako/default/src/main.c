#include <match/default_inputs.h>
#include <match/runtime.h>

// target specific inlcudes
% for inc_h in target.include_list:
#include <${inc_h}.h>
% endfor
% if golden_cpu_model:
#define GOLDEN_CHECK_BENCH_ITERATIONS 100
% endif

int main(int argc,char** argv){
    // target specific inits
    % for init_func in target.init_funcs:
    ${init_func}();
    % endfor
    
    match_runtime_ctx match_ctx;

    % if runtime=="generative":
    % for inp_name,inp in match_inputs.items():
    ${inp["c_type"]} ${inp_name} = 1;
    % endfor
    
    % for out_name,out in match_outputs.items():
    ${out["c_type"]} ${out_name} = 0;
    % endfor
    
    match_generative_runtime(
        % for inp_name in match_inputs.keys():
        &${inp_name},
        % endfor
        % for out_name in match_outputs.keys():
        &${out_name},
        % endfor
        % for dyn_dim in dynamic_dims.keys():
        1,
        % endfor
        &match_ctx
    );

    % else:
    % for out_name,out in match_outputs.items():
    ${out["c_type"]}* ${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % if golden_cpu_model:
    ${out["c_type"]}* golden_check_${out_name}_pt = ${target.alloc_fn}(sizeof(${out["c_type"]}) * ${out["prod_shape"]});
    % endif
    % endfor

    match_${"golden_check_" if golden_cpu_model else ""}${runtime}_runtime(
        % for out_name in match_outputs.keys():
        ${out_name}_pt,
        % endfor
        % if golden_cpu_model:
        % for out_name in match_outputs.keys():
        golden_check_${out_name}_pt,
        % endfor
        GOLDEN_CHECK_BENCH_ITERATIONS,
        % endif
        &match_ctx);
    % endif
    
    % for out_name in match_outputs.keys():
    % if golden_cpu_model:
    ${target.free_fn}(golden_check_${out_name}_pt);
    % endif
    ${target.free_fn}(${out_name}_pt);
    % endfor
    // target specific cleaning functions
    % for clean_func in target.clean_funcs:
    ${clean_func}();
    % endfor
    return 0;
}