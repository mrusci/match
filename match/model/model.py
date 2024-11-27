

from math import prod
import os
from pathlib import Path
import subprocess
from typing import Dict, List

import mako
import numpy as np
from match.model.dynamic_dim import DynamicDim
from match.utils import save_all_relay,add_save_relay,reset_relay_list,reset_output_path,set_output_path,reset_schedules,save_all_schedules
from match.driver.driver import driver
from mako.template import Template

from match.utils.utils import numpy_dtype_to_c_type
import tvm

class MatchModel:

    def __init__(self,relay_mod,relay_params,dynamic:bool=False,dyn_is_max:bool=False,dynamic_sizes:Dict={}):
        self.relay_mod = relay_mod
        self.relay_params = relay_params
        self.dynamic = dynamic
        self.dynamic_sizes = dynamic_sizes
        self.dyn_is_max = dyn_is_max
        self.name = "default" if (not self.dynamic or self.dyn_is_max) else "_".join(f"{dyn_name.replace(' ', '_')}_{dyn_val}" for dyn_name, dyn_val in dynamic_sizes.items())
    

    def get_path(self,out_path):
        return str(Path(out_path).absolute())+"/"+self.name+"_buildtmp"


    def compile_model(self,target,out_path):
        reset_output_path()
        reset_relay_list()
        reset_schedules()
        model_path = self.get_path(out_path=out_path)
        set_output_path(model_path)
        #breakpoint()
        driver(self.relay_mod, self.relay_params, target=target,output_path=model_path,mod_name=self.name)
        #self.rename_lib(out_path=out_path)
        save_all_relay()
        save_all_schedules()

    def rename_lib(self,out_path):
        pass

    def remove_static_app(self):
        pass

    def move_static_app_to(self,out_path):
        build_dir = self.get_path(out_path=out_path)
        abs_out_path = str(Path(out_path).absolute())
        
        # create codegen if it doesn't exist
        if not Path(abs_out_path+"/codegen").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/codegen")
        if not Path(abs_out_path+"/models").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/models")
        
        def create_mod_dir_and_mv(rm_dirlist):
            subprocess.getoutput(f"mkdir {abs_out_path}/models/{self.name}")
            for ext_type in ["relay","log"]:
                subprocess.getoutput(f"mkdir {abs_out_path}/models/{self.name}/match_{ext_type}")
                subprocess.getoutput(f"cp {build_dir}/*.{ext_type} {abs_out_path}/models/{self.name}/match_{ext_type}")
                subprocess.getoutput(f"rm {build_dir}/*.{ext_type}")
            # parameters
            subprocess.getoutput(f"mv {build_dir}/parameters {abs_out_path}/models/{self.name}/parameters")
            # codegen
            subprocess.getoutput(f"mv {build_dir}/codegen/host {abs_out_path}/codegen/{self.name}")
            # rm stuff now
            for rm_dir in rm_dirlist:
                subprocess.getoutput(f"rm {build_dir}/{rm_dir} -r")
            subprocess.getoutput(f"mkdir {abs_out_path}/models/{self.name}/other_metadata")
            subprocess.getoutput(f"cp {build_dir}/* {abs_out_path}/models/{self.name}/other_metadata")
            subprocess.getoutput(f"rm {build_dir} -r")

        def move_final_relay():
            subprocess.getoutput(f"mv {build_dir}/src/{self.name}.relay {build_dir}/{self.name}.relay")
        
        if (not self.dynamic) or (self.dyn_is_max):
            # move app
            move_final_relay()
            # include src runtime
            for static_dir in ["include","src","runtime"]:
                subprocess.getoutput(f"mv {build_dir}/{static_dir} {abs_out_path}/{static_dir}")
            create_mod_dir_and_mv(rm_dirlist=["codegen","templates"])
        else:
            # remove all the static stuff and move the codegen in the overall build folder
            move_final_relay()
            create_mod_dir_and_mv(rm_dirlist=["codegen","templates","include","src","runtime"])



def build_runtime(static_models:List[MatchModel]=[],dynamic_dims:Dict[str,DynamicDim]={},match_inputs=None,match_outputs=None,runtime:str="default",out_path:str="/build"):
    abs_out_path = str(Path(out_path).absolute())
    temp_args = {
        "generative_models":{model.name:model for model in static_models},
        "models":{model.name:model for model in static_models},
        "dynamic_dims":dynamic_dims,
        "outputs":match_outputs,
        "inputs":match_inputs,
        "runtime":runtime,
    }
    with open(abs_out_path+"/src/match_runtime.c","w") as run_file:
        run_file.write(Template(filename=os.path.dirname(__file__)+"/../codegen/template/lib/match_runtime_template.c").render(**temp_args))
    with open(abs_out_path+"/include/match_runtime.h","w") as run_file:
        run_file.write(Template(filename=os.path.dirname(__file__)+"/../codegen/template/lib/match_runtime_template.h").render(**temp_args))

def get_match_inputs_and_outputs(static_models:List[MatchModel]=[]):
    mod_checked_types = tvm.relay.transform.InferType()(static_models[0].relay_mod)
    func=mod_checked_types["main"]
    relay_inputs=func.params
    relay_out_types=func.checked_type.arg_types[1:]
    match_inputs={inp_.name_hint:{"name":inp_.name_hint,"c_arr_size":int(prod(inp_.type_annotation.shape)*np.dtype(inp_.type_annotation.dtype).itemsize),"c_type":numpy_dtype_to_c_type(inp_.type_annotation.dtype),"prod_shape":int(prod(inp_.type_annotation.shape)),"shape":[int(sh) for sh in inp_.type_annotation.shape], "c_arr_values":"{"+str([1 for _ in range(int(prod(inp_.type_annotation.shape)))])[1:-1]+"}"} for inp_ in relay_inputs}
    match_outputs = {f"output{idx if len(relay_out_types)>1 else ''}":{"name":f"output{idx if len(relay_out_types)>1 else ''}","c_arr_size":int(prod(out.shape)*np.dtype(out.dtype).itemsize),"c_type":numpy_dtype_to_c_type(out.dtype),"prod_shape":int(prod(out.shape)),"shape":[int(sh) for sh in out.shape],} for idx,out in enumerate(relay_out_types)}
    return match_inputs,match_outputs