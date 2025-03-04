

import json
from math import prod
import os
from pathlib import Path
import subprocess

import numpy as np
from numpy import typing as npt
from match.compile.c_graph import MatchCompilerCGraph
from match.model.dynamic_dim import DynamicDim
from match.model.runtime import MatchTVMGraphRuntime
from match.relay.get_relay import get_dyn_relay_from, get_relay_from
from match.utils import save_all_relay,add_save_relay,reset_relay_list,reset_output_path,\
                        set_output_path,reset_schedules,save_all_schedules
from match.compile.c_aot import MatchCompilerCAoT
from mako.template import Template

from match.utils.utils import c_friendly_npvalue, get_random_np_array, numpy_dtype_to_c_type, set_model_name
import tvm
from tvm import relay

EXCUTOR_COMPILER_CLS = {"aot":MatchCompilerCAoT, "graph":MatchCompilerCGraph}    

class MatchModel:

    def __init__(self, relay_mod=None, relay_params=None, filename="model.onnx", params_filename="params.data",
                 model_type="onnx", model_name="default", golden_cpu_model=True, benchmark_model=True, executor="aot",
                 default_inputs=None, is_model_dynamic=False, dynamic_algorithm="cuts", dynamic_dims=None):
        self.relay_mod = relay_mod
        self.relay_params = relay_params
        self.filename = filename
        self.params_filename = params_filename
        self.model_type = model_type
        self.model_name = model_name
        self.golden_cpu_model = golden_cpu_model
        self.benchmark_model = benchmark_model
        # how to compile the model
        self.executor = "aot"
        self.compilation_cls = EXCUTOR_COMPILER_CLS["aot"]
        if executor in EXCUTOR_COMPILER_CLS:
            self.executor = executor
            self.compilation_cls = EXCUTOR_COMPILER_CLS[self.executor]
        self.default_inputs = default_inputs
        # dynamic model params
        self.is_model_dynamic = is_model_dynamic
        self.dynamic_algorithm = dynamic_algorithm
        self.dynamic_dims = dynamic_dims
        # self.name = model_name if (not self.dynamic or self.dyn_is_max) else "_".join(f"{dyn_name.replace(' ', '_')}_{dyn_val}" for dyn_name, dyn_val in dynamic_sizes.items())    
        # may be used for dynamic models in case c executor is used
        self.other_models = dict()

    @staticmethod
    def get_path(out_path,model_name):
        return str(Path(out_path).absolute())+"/"+model_name+"_buildtmp"

    @staticmethod
    def set_model(out_path, model_name):
        reset_output_path()
        reset_relay_list()
        reset_schedules()
        model_path = MatchModel.get_path(out_path=out_path, model_name=model_name)
        set_output_path(model_path)
        set_model_name(model_name)
        return model_path

    @staticmethod
    def save_model_logs():
        save_all_relay()
        save_all_schedules()

    def compile(self, target, out_path):
        # reset the session
        if self.relay_mod is None:
            if self.is_model_dynamic:
                # this function returns the basic relay mod and relay params to run and a bunch of other models
                # this may be helpful if using a dynamic model and the goal is the one of cutting it into several
                # static models, where the basic one will be the one with max size and the others are the cuts
                self.relay_mod, self.relay_params, self.other_models = get_dyn_relay_from(self.model_type, self.filename, self.params_filename, self.model_name, self.dynamic_dims, self.dynamic_algorithm)
            else:
                self.relay_mod, self.relay_params = get_relay_from(self.model_type, self.filename, self.params_filename)
        
        if len(self.other_models)>0:
            for model_name, model in self.other_models.items():
                model_path = MatchModel.set_model(out_path=out_path, model_name=model_name)
                comp =  self.compilation_cls(
                            model[0], model[1],
                            target = target,
                            build_dir = model_path,
                            mod_name = model_name
                        )
                comp.tvm_compile(fusion=True)
                MatchModel.save_model_logs()
                MatchModel.gen_model_runtime_and_move_model(target=target, out_path=out_path,
                                                            model_name=model_name, executor=self.executor)

        # compile the golden cpu to check if used
        if self.golden_cpu_model:
            target_disabled_modules = target.disabled_exec_modules
            new_disabled_modules = []
            for ex_mod in target.exec_modules_dict:
                new_disabled_modules.append(ex_mod)
            target.disabled_exec_modules = new_disabled_modules
            model_path = MatchModel.set_model(out_path=out_path, model_name=self.model_name+"_golden_cpu")
            comp =  self.compilation_cls(
                        self.relay_mod, self.relay_params,
                        target = target,
                        build_dir = model_path,
                        mod_name = self.model_name+"_golden_cpu"
                    )
            comp.tvm_compile(fusion=True)
            MatchModel.save_model_logs()
            MatchModel.gen_model_runtime_and_move_model(target=target, out_path=out_path,
                                                        model_name=self.model_name+"_golden_cpu", executor=self.executor)
            target.disabled_exec_modules = target_disabled_modules

        # compile the model
        model_path = MatchModel.set_model(out_path=out_path, model_name=self.model_name)
        comp =  self.compilation_cls(
                    self.relay_mod, self.relay_params,
                    target = target,
                    build_dir = model_path,
                    mod_name = self.model_name
                )
        comp.tvm_compile(fusion=True)
        MatchModel.save_model_logs()
        MatchModel.gen_model_runtime_and_move_model(target=target, out_path=out_path,
                                                    model_name=self.model_name, executor=self.executor)
        self.model_runtime(target=target,out_path=out_path)
    
    def model_runtime(self, target=None, out_path:str="/build"):
        abs_out_path = str(Path(out_path).absolute())
        match_inputs, match_outputs = self.get_match_inputs_and_outputs()
        temp_args = {
            "match_model":self,
            "outputs":match_outputs,
            "inputs":match_inputs,
            "all_model_names":[self.model_name]+([] if not self.golden_cpu_model else [self.model_name+"_golden_cpu"]) + [key_ for key_ in self.other_models],
            "target":target,
        }
        if not Path(abs_out_path+f"/src/{self.model_name}").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/src/{self.model_name}")
        if not Path(abs_out_path+f"/include/{self.model_name}").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/include/{self.model_name}")
        with open(abs_out_path+f"/src/{self.model_name}/runtime.c","w") as run_file:
            run_file.write(Template(filename=os.path.dirname(__file__)+"/../libs/c/mako/match/src/runtime.c").render(**temp_args))
        with open(abs_out_path+f"/include/{self.model_name}/runtime.h","w") as run_file:
            run_file.write(Template(filename=os.path.dirname(__file__)+"/../libs/c/mako/match/include/runtime.h").render(**temp_args))
        
    @staticmethod
    def gen_model_runtime_and_move_model(target=None, out_path:str="./match_out",
                                         model_name:str="default", executor:str="graph",
                                         ):
        build_dir = MatchModel.get_path(out_path=out_path,model_name=model_name)
        abs_out_path = str(Path(out_path).absolute())
        if executor=="graph":
            # setup the dir structure as if it is an aot codegen
            subprocess.getoutput(f"mkdir {build_dir}/codegen")
            subprocess.getoutput(f"mkdir {build_dir}/codegen/host")
            subprocess.getoutput(f"mkdir {build_dir}/codegen/host/include")
            subprocess.getoutput(f"mkdir {build_dir}/codegen/host/src")
            subprocess.getoutput(f"tar -xvf {build_dir}/mod.tar -C {build_dir}/codegen/host/src")
            # read the json of the mod and the params to build the runtime
            mod_info = {}
            with open(f"{build_dir}/mod.json","r") as mod_file:
                mod_info = json.load(mod_file)
            if len(mod_info)==0:
                raise FileNotFoundError()
            params_bytes=bytes("","utf8")
            params = None
            with open(f"{build_dir}/mod.params","rb") as params_file:
                params_bytes=params_file.read()
            params=relay.load_param_dict(params_bytes)
            # now with both params and mod info generate a runtime considering memory planning etc.
            graph_runtime = MatchTVMGraphRuntime(target=target, mod_info=mod_info, params=params, model_name=model_name, out_path=build_dir)
            graph_runtime_template_data = graph_runtime.generate()
            try:
                with open(f"{build_dir}/codegen/host/src/graph_runtime.c","w") as run_file:
                    run_file.write(Template(filename = os.path.dirname(__file__)+"/../libs/c/mako/match/src/graph_runtime.c").render(**graph_runtime_template_data))
                # with open(f"{build_dir}/codegen/host/include/graph_runtime.c","w") as run_file:
                    # run_file.write(Template(filename = os.path.dirname(__file__)+"/../libs/c/mako/match/include/graph_runtime.h").render(**graph_runtime_template_data))
            except Exception as e:
                print(f"[TEMPLATE WRITER] Error processing graph runtime template")
                raise e
            subprocess.getoutput(f"rm {build_dir}/mod.tar")
        # create codegen if it doesn't exist
        if not Path(abs_out_path+"/codegen").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/codegen")
        if not Path(abs_out_path+"/models").is_dir():
            subprocess.getoutput(f"mkdir {abs_out_path}/models")
        
        def create_mod_dir_and_mv(rm_dirlist):
            subprocess.getoutput(f"mkdir {abs_out_path}/models/{model_name}")
            for ext_type in ["relay","log"]:
                subprocess.getoutput(f"mkdir {abs_out_path}/models/{model_name}/{ext_type}")
                subprocess.getoutput(f"cp {build_dir}/*.{ext_type} {abs_out_path}/models/{model_name}/{ext_type}")
                subprocess.getoutput(f"rm {build_dir}/*.{ext_type}")
            # parameters
            subprocess.getoutput(f"mv {build_dir}/parameters {abs_out_path}/models/{model_name}/parameters")
            # codegen
            subprocess.getoutput(f"mv {build_dir}/codegen/host {abs_out_path}/codegen/{model_name}")
            # nodes
            if Path(build_dir+f"/src/nodes/{model_name}").is_dir():
                if not Path(abs_out_path+"/src/nodes").is_dir():
                    subprocess.getoutput(f"mkdir {abs_out_path}/src/nodes")
                if not Path(abs_out_path+"/include/nodes").is_dir():
                    subprocess.getoutput(f"mkdir {abs_out_path}/include/nodes")
                subprocess.getoutput(f"mv {build_dir}/src/nodes/{model_name} {abs_out_path}/src/nodes/{model_name}")
                subprocess.getoutput(f"mv {build_dir}/include/nodes/{model_name} {abs_out_path}/include/nodes/{model_name}")
            # model
            subprocess.getoutput(f"mv {build_dir}/src/{model_name} {abs_out_path}/src/{model_name}")
            subprocess.getoutput(f"mv {build_dir}/include/{model_name} {abs_out_path}/include/{model_name}")
            # rm stuff now
            for rm_dir in rm_dirlist:
                subprocess.getoutput(f"rm {build_dir}/{rm_dir} -r")
            subprocess.getoutput(f"mkdir {abs_out_path}/models/{model_name}/metadata")
            subprocess.getoutput(f"cp {build_dir}/* {abs_out_path}/models/{model_name}/metadata")
            subprocess.getoutput(f"rm {build_dir} -r")

        def move_final_relay():
            subprocess.getoutput(f"mv {build_dir}/src/{model_name}.relay {build_dir}/{model_name}.relay")
        
        if not Path(f"{abs_out_path}/runtime").exists():
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
    
    def get_match_inputs_and_outputs(self):
        # infer types and get info about the relay graph
        mod_checked_types = tvm.relay.transform.InferType()(self.relay_mod)
        func=mod_checked_types["main"]
        relay_inputs=func.params
        if isinstance(func.checked_type.ret_type,tvm.relay.TupleType):
            relay_out_types = [ret_type for ret_type in func.checked_type.ret_type.fields]
        else:
            relay_out_types=[func.checked_type.ret_type]
        default_inputs = self.default_inputs
        if default_inputs is not None:
            default_inputs = [c_friendly_npvalue(inp_) for inp_ in default_inputs]
        else:
            # generate random inputs
            np.random.seed(0)
            default_inputs = [c_friendly_npvalue(get_random_np_array(dtype=inp_.type_annotation.dtype,
                                                                     shape=tuple([int(v) for v in inp_.type_annotation.shape])))
                                                for inp_ in relay_inputs]
        # format inputs and outputs data structure as C templates expect
        match_inputs={
            inp_.name_hint:{
                "name":inp_.name_hint,
                "c_arr_size":int(prod(inp_.type_annotation.shape)),
                "c_type":numpy_dtype_to_c_type(inp_.type_annotation.dtype),
                "prod_shape":int(prod(inp_.type_annotation.shape)),
                "shape":[int(sh) for sh in inp_.type_annotation.shape],
                "dims":[int(sh) for sh in inp_.type_annotation.shape],
                "c_arr_values":default_inputs[idx],
            } for idx,inp_ in enumerate(relay_inputs)
        }
        match_outputs = {
            f"output{idx if len(relay_out_types)>1 else ''}":{
                "name":f"output{idx if len(relay_out_types)>1 else ''}",
                "c_arr_size":int(prod(out.shape)),
                "c_type":numpy_dtype_to_c_type(out.dtype),
                "prod_shape":int(prod(out.shape)),
                "shape":[int(sh) for sh in out.shape],
                "dims":[int(sh) for sh in out.shape],
            } for idx,out in enumerate(relay_out_types)
        }
        return match_inputs,match_outputs