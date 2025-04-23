# test file for ODL test
# derived from: test/test.py
# FIXME: must


import argparse
import os
from pathlib import Path
from typing import List


from targets.pulp_open import PulpOpen
from targets.default import DefaultExample
from targets.GAP9 import GAP9
from utils import get_default_inputs
from microbench import get_microbench_mod, get_network_single_nodes
from tvm import relay

import match
from match.model.model import MatchModel
import onnx

TEST_TARGET = {
    "pulp_open": PulpOpen,
    "GAP9": GAP9,
    "default": DefaultExample,
}

def save_model_and_params(mod,params):
    if not Path(os.path.dirname(__file__)+"/models").is_dir():
        Path(os.path.dirname(__file__)+"/models").mkdir()
    if not Path(os.path.dirname(__file__)+"/models/last_model").is_dir():
        Path(os.path.dirname(__file__)+"/models/last_model").mkdir()
    with open(str(Path(os.path.dirname(__file__)+"/models/last_model/model_graph.relay").absolute()),"w") as mod_file:
        mod_file.write(relay.astext(mod))
    with open(str(Path(os.path.dirname(__file__)+"/models/last_model/model_params.txt").absolute()),"wb") as par_file:
        par_file.write(relay.save_param_dict(params=params))

def run_nodes_of_network(target_name: str="default", network: str="conv", output_path: str="./builds/last_build", executor: str="aot",
                        golden_cpu_model: bool=False, input_files: List[str]=[], min_input_val=None, max_input_val=None):
    single_node_mod_params = get_network_single_nodes(network)
    #define HW Target inside match
    if target_name not in TEST_TARGET:
        raise Exception(f"{target_name} target is not available, the targets available are {[k for k in TEST_TARGET.keys()]}")
    target = TEST_TARGET[target_name]()
    if not Path(os.path.dirname(__file__)+f"/builds/{network}").is_dir():
        Path(os.path.dirname(__file__)+f"/builds/{network}").mkdir()
    idx_inp_file = 0
    for node in single_node_mod_params:
        node_name, mod, params = node[0], node[1], node[2]
        output_path = str(Path(f"{output_path}/{network}/{node_name}").absolute())
        default_inputs = get_default_inputs(mod=mod, input_files=input_files[idx_inp_file:],
                                                min_input_val=min_input_val, max_input_val=max_input_val)
        match.match(
            model=MatchModel(
                relay_mod=mod, relay_params=params,
                model_name=network+f"_{node_name}", executor=executor,
                default_inputs=default_inputs,
                golden_cpu_model=golden_cpu_model,
            ),
            target=target,
            output_path=output_path
        )
        if len(input_files)>0:
            idx_inp_file += len(default_inputs)


def run_microbench(target_name: str="default", microbench: str="conv", output_path: str="./builds/last_build", executor: str="aot",
                   golden_cpu_model: bool=False, input_files: List[str]=[], min_input_val=None, max_input_val=None):
    mod,params = get_microbench_mod(microbench)
    save_model_and_params(mod=mod,params=params)
    #define HW Target inside match
    if target_name not in TEST_TARGET:
        raise Exception(f"{target_name} target is not available, the targets available are {[k for k in TEST_TARGET.keys()]}")
    target = TEST_TARGET[target_name]()

    match.match(
        model=MatchModel(
           relay_mod=mod, relay_params=params,
           model_name=microbench, executor=executor,
           default_inputs=get_default_inputs(mod=mod, input_files=input_files,
                                             min_input_val=min_input_val, max_input_val=max_input_val),
           golden_cpu_model=golden_cpu_model,
        ),
        target=target,
        output_path=output_path
    )

def run_model(target_name: str="pulp_platform", model: str="keyword_spotting", output_path: str="./builds/last_build", executor: str="aot",
              golden_cpu_model: bool=False, input_files: List[str]=[], min_input_val=None, max_input_val=None):
    #define HW Target inside match
    if target_name not in TEST_TARGET:
        raise Exception(f"{target_name} target is not available, the targets available are {[k for k in TEST_TARGET.keys()]}")
    target = TEST_TARGET[target_name]()
    onnx_model_filepath = os.path.dirname(__file__)+"/models/"+model+".onnx"
    onnx_model = onnx.load(onnx_model_filepath)
    try:
        onnx.checker.check_model(onnx_model)
        mod, _ = relay.frontend.from_onnx(onnx_model)
    except Exception as e:
        # this is a line to sanitize plinio-typed onnx
        # cartoonnyx.cartoonnyx.plinio.sanitize_to_mps(onnx_model_filepath)
        raise e
    
    match.match(
        model=MatchModel(
           filename=onnx_model_filepath,
           model_type="onnx",
           model_name=model, executor=executor,
           default_inputs=get_default_inputs(mod=mod, input_files=input_files,
                                             min_input_val=min_input_val, max_input_val=max_input_val),
           golden_cpu_model=golden_cpu_model
        ),
        target=target,
        output_path=output_path
    )

def run_model_relay(tvmvir_mod_filename, tvmvir_params_filename, 
                    target_name: str="pulp_platform", model: str="keyword_spotting", # FIXME change me!!!!
                    output_path: str="./builds/last_build", executor: str="aot",
                    golden_cpu_model: bool=False, input_files: List[str]=[], 
                    min_input_val=None, max_input_val=None ):
    
    #define HW Target inside match
    if target_name not in TEST_TARGET:
        raise Exception(f"{target_name} target is not available, the targets available are {[k for k in TEST_TARGET.keys()]}")
    target = TEST_TARGET[target_name]()

    match.match(
        model=MatchModel(
           filename=tvmvir_mod_filename,
           params_filename = tvmvir_params_filename,
           model_type="relay_odl",
           model_name=model, executor=executor,
#           default_inputs=get_default_inputs(mod=mod, input_files=input_files,
#                                             min_input_val=min_input_val, max_input_val=max_input_val),
           golden_cpu_model=golden_cpu_model
        ),
        target=target,
        output_path=output_path
    )

def run_relay_saved_model_at(target_name: str="pulp_platform", mod_file: str="./models/last_model/model_graph.relay",
                             params_file: str="./models/last_model/model_params.txt", output_path: str="./builds/last_build",
                             executor: str="aot", golden_cpu_model: bool=False, input_files: List[str]=[],
                             min_input_val=None, max_input_val=None):
    #define HW Target inside match
    if target_name not in TEST_TARGET:
        raise Exception(f"{target_name} target is not available, the targets available are {[k for k in TEST_TARGET.keys()]}")
    target = TEST_TARGET[target_name]()
    # just to set the desired inputs and check that there is an actual model there
    with open(mod_file,"r") as mod_file:
        mod_text=mod_file.read()
    mod=relay.fromtext(mod_text)

    match.match(
        model=MatchModel(
           filename=mod_file, params_filename=params_file,
           model_type="relay", model_name="default", executor=executor,
           default_inputs=get_default_inputs(mod=mod, input_files=input_files,
                                             min_input_val=min_input_val, max_input_val=max_input_val),
           golden_cpu_model=golden_cpu_model
        ),
        target=target,
        output_path=output_path
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compile_last_model",
        dest="compile_last_model",
        action="store_true",
        help="Compile the last model saved",
    )
    parser.add_argument(
        "--microbench",
        dest="microbench",
        type=str,
        default="conv",
        help="Microbench test to compile, such as 2d Convolution example node, that contains a Conv2d, a bias add and a ReLU operation",
    )

    parser.add_argument(
        "--model",
        dest="model",
        default="",
        type=str,
        help="Model name to compile, models are available in the test/models directory, they must be in ONNX format."
    )

    parser.add_argument(
        "--network",
        dest="network",
        default="",
        type=str,
        help="Network name to compile, networks are compiled taking each single node separetely."
    )

    parser.add_argument(
        "--executor",
        dest="executor",
        default="aot",
        type=str,
        choices=["aot","graph"],
        help="Choose the executor to use within MATCH",
    )

    parser.add_argument(
        "-g",
        "--golden",
        dest="golden",
        action="store_true",
        help="Compile also the golden cpu model",
    )

    parser.add_argument(
        "-t",
        "--target",
        dest="target",
        default="pulp_open",
        type=str,
        choices=["default", "pulp_open", "GAP9"],
        help="Choose the target to run the test on"
    )

    parser.add_argument(
        "--input_files",
        dest="input_files",
        type=str,
        nargs='+',
        help="List of input files to use as inputs for the model"
    )

    parser.add_argument(
        "--min_input_val",
        dest="min_input_val",
        type=float,
        help="Minimum value for input data"
    )

    parser.add_argument(
        "--max_input_val",
        dest="max_input_val",
        type=float,
        help="Maximum value for input data"
    )

    parser.add_argument(
        "--relay_model",
        type=str,
        help="Path to the saved model"
    )
    parser.add_argument(
        "--relay_params_filename",
        type=str,
    )
    args = parser.parse_args()
    if not Path(os.path.dirname(__file__)+"/builds").is_dir():
        Path(os.path.dirname(__file__)+"/builds").mkdir()
    if args.compile_last_model and Path(os.path.dirname(__file__)+"/models/last_model").is_dir() and \
        Path(os.path.dirname(__file__)+"/models/last_model/model_graph.relay").exists() and \
            Path(os.path.dirname(__file__)+"/models/last_model/model_params.txt").exists():
        run_relay_saved_model_at(target_name=args.target,
                                 mod_file=str(Path(os.path.dirname(__file__)+"/models/last_model/model_graph.relay").absolute()),
                                 params_file=str(Path(os.path.dirname(__file__)+"/models/last_model/model_params.txt").absolute()),
                                 output_path=str(Path(os.path.dirname(__file__)+"/builds/last_build").absolute()),
                                 input_files=args.input_files,
                                 min_input_val=args.min_input_val,
                                 max_input_val=args.max_input_val
                                 )
    else:
        if args.relay_model!="":
            run_model_relay( args.relay_model, args.relay_params_filename, 
                      target_name=args.target,
                      model="test_bp_tvm",
                      output_path=str(Path(os.path.dirname(__file__)+"/builds/last_build").absolute()),
                      executor=args.executor,
                      golden_cpu_model=args.golden,
                      input_files=args.input_files,
                      min_input_val=args.min_input_val,
                      max_input_val=args.max_input_val
                      )
        elif args.model!="":
            run_model(target_name=args.target,
                      model=args.model,
                      output_path=str(Path(os.path.dirname(__file__)+"/builds/last_build").absolute()),
                      executor=args.executor,
                      golden_cpu_model=args.golden,
                      input_files=args.input_files,
                      min_input_val=args.min_input_val,
                      max_input_val=args.max_input_val
                      )
        elif args.network!="":
            run_nodes_of_network(target_name=args.target,
                                network=args.network,
                                output_path=str(Path(os.path.dirname(__file__)+"/builds/last_build").absolute()),
                                executor=args.executor,
                                golden_cpu_model=args.golden,
                                input_files=args.input_files,
                                min_input_val=args.min_input_val,
                                max_input_val=args.max_input_val
                                )
        else:
            run_microbench(target_name=args.target,
                           microbench=args.microbench,
                           output_path=str(Path(os.path.dirname(__file__)+"/builds/last_build").absolute()),
                           executor=args.executor,
                           golden_cpu_model=args.golden,
                           input_files=args.input_files,
                           min_input_val=args.min_input_val,
                           max_input_val=args.max_input_val
                           )