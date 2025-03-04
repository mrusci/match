from math import floor
from typing import Any, Dict, List

from match.dim.dim import MatchDim, MatchTiledDim
from match.node.node import MatchNode
from match.opt.engine import ScheduleEngine
from match.schedule.block import MatchBlock
from match.schedule.mem_transfer import MatchMemTransfer
from match.schedule.loop import MatchLoop
from match.schedule.schedule import MatchSchedule
from match.target.exec_module import ExecModule
from match.target.memory_inst import MemoryInst, PortConnection
# zigzag imports
from match.tensor.tensor import MatchTensorTile
from zigzag import api
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.opt.temporal.loma.engine import NoValidLoopOrderingFoundException

class ZigZagEngine(ScheduleEngine):
    def __init__(self,exec_module:ExecModule=None,pattern_name:str="",match_node:MatchNode=None):
        super(ZigZagEngine, self).__init__(exec_module=exec_module,pattern_name=pattern_name,match_node=match_node)
        self.lpf_limit=13
        self.debug_node_schedule=False
        self.zigzag_temporal_mapping=dict()

    def transform_schedule_for_engine(self):
        # TODO: get this params correctly
        o_intermediate_prec,o_prec,first_inp_prec,second_inp_prec = 32,8,8,8
        conv = "conv1d" in self.match_node.ops_occurrences or "conv2d" in self.match_node.ops_occurrences
        conv2d = "conv2d" in self.match_node.ops_occurrences
        conv2d_is_dw = conv2d and self.match_node.ops["conv2d"].depthwise
        conv1d_is_dw = conv and (not conv2d) and self.match_node.ops["conv1d"].depthwise
        add = (not conv2d) and ("add" in self.match_node.ops_occurrences)
        if conv:
            if conv2d:
                strides = self.match_node.ops["conv2d"].strides
                dilations = self.match_node.ops["conv2d"].dilation
                padding = self.match_node.ops["conv2d"].padding
                k_size = self.match_node.ops["conv2d"].kernel_size
            else:
                # get data
                strides = self.match_node.ops["conv1d"].strides
                dilations = self.match_node.ops["conv1d"].dilation
                padding = self.match_node.ops["conv1d"].padding
                k_size = self.match_node.ops["conv1d"].kernel_size
                # transform it like conv2d
                strides = strides + (1,)
                dilations = dilations + (1,)
                # as if it was height only
                padding = (padding[0], 0, padding[1], 0)
                k_size = k_size + (1,)
        else:
            strides = (1,1)
            dilations = (1,1)
            padding = (0,0,0,0)
            k_size = (1,1)
        padding = {"IY":(padding[0],padding[2]),"IX":(padding[1],padding[3])}
        inp_tensors_names = [n for n in self.match_node.var_tensors.keys()]
        out_tensors_names = [n for n in self.match_node.output_tensors.keys()]
        inp_dims = [d.size for d in self.match_node.var_tensors[inp_tensors_names[0]].dims]
        out_dims = [d.size for d in self.match_node.output_tensors[out_tensors_names[0]].dims]
        o_n,o_c,o_h,o_w = (inp_dims[0],out_dims[1]) + ((out_dims[2],out_dims[3]) if len(out_dims)>3 else (out_dims[2],) if len(out_dims)>2 else (1,1))
        i_h,i_w = (inp_dims[2],inp_dims[3]) if len(inp_dims)>3 else (inp_dims[2],) if len(inp_dims)>2 else (1,1)
        w_cin, w_ks_h, w_ks_w = inp_dims[1], k_size[0], k_size[1]
        kernel_size = (w_ks_h, w_ks_w)
        dimension_relations = [
            f"ix={strides[1]}*ox+{dilations[1]}*fx",
            f"iy={strides[0]}*oy+{dilations[0]}*fy",
        ]
        equation = "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]" if conv2d and not conv2d_is_dw else "O[b][k][oy][ox]+=X[b][k][oy][ox]*Y[b][k][oy][ox]" if add else "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][k][iy][ix]"
        loop_dim_size = {
            "B": o_n,
            "K": o_c,
            "C": w_cin,
            "OY": o_h,
            "OX": o_w,
            "FY": kernel_size[0],
            "FX": kernel_size[1],
        }
        operand_precision = {
            "O": o_intermediate_prec,
            "O_final": o_prec,
            "I" if not add else "X": first_inp_prec,
            "W" if not add else "Y": second_inp_prec,
        }
        pr_loop_dim_size = {"IY": i_h, "IX": i_w}
        operand_source={"W": [], "I": []} if not add else {"X":[], "Y":[]}
        constant_operands = ["W"] if not add else []
        operand_source_dimension_mapping={"I": {"IX": "OX", "IY": "OY"}} if not add else {
            "X": {"IX": "OX", "IY": "OY", "C": "K"},
            "Y": {"IX": "OX", "IY": "OY", "C": "K"},
        }
        if conv2d_is_dw or conv1d_is_dw:
            operand_source_dimension_mapping["I"]["C"]="K"
        self.workload={
            1: {
                "operator_type": self.pattern_name,
                "equation": equation,
                "dimension_relations": dimension_relations,
                "loop_dim_size": loop_dim_size,
                "operand_precision": operand_precision,
                "pr_loop_dim_size": pr_loop_dim_size,
                "padding": padding,
                "strides": strides,
                "operand_source": operand_source,
                "constant_operands": constant_operands,
                'operand_source_dimension_mapping': operand_source_dimension_mapping,
            }
        }

    def generate_accelerator(self,platform_memories:List[MemoryInst]=[],optimal_spatial_mapping:List[Any]=[]):

        def zigzag_port_def(port:PortConnection,read_def:bool=True):
                    if port.reading_port is None:
                        return None
                    elif read_def:
                        return f"{port.reading_port['type']}_port_{port.reading_port['number']}"
                    return f"{port.writing_port['type']}_port_{port.writing_port['number']}"
        def rename_operands(operands: List[str]=[]):
            return ["O" if op=="O" else "I1" if op=="I" or op=="X" else "I2" for op in operands]
        
        def get_memory_hierarchy(multiplier_array,platform_memories:List[MemoryInst]=[]):
            """Memory hierarchy variables"""
            """ size=#bit, bw=#bit"""
            memory_hierarchy_graph = MemoryHierarchy(operational_array=multiplier_array)
            for plat_mem in platform_memories:
                mem_inst = MemoryInstance(
                    name = plat_mem.name,
                    size = floor(( plat_mem.k_bytes * 1024 - plat_mem.buffer_for_layer_func(self.match_node,self.pattern_name,self.match_node.specific_pattern))* 8),
                    r_bw=floor(plat_mem.r_bw),
                    w_bw=floor(plat_mem.w_bw),
                    r_cost=100,
                    w_cost=110,
                    area=0,
                    r_port=floor(plat_mem.r_ports),
                    w_port=floor(plat_mem.w_ports),
                    rw_port=floor(plat_mem.rw_ports),
                    latency=1, # TODO: Unused in ZigZag, should be > 1
                    double_buffering_support=plat_mem.double_buffering_support
                )
                port_alloc=tuple([
                    {
                        "fh":zigzag_port_def(port=plat_mem.used_ports[operand][0],read_def=False),
                        "tl":zigzag_port_def(port=plat_mem.used_ports[operand][0],read_def=True),
                        "fl":zigzag_port_def(port=plat_mem.used_ports[operand][1],read_def=False),
                        "th":zigzag_port_def(port=plat_mem.used_ports[operand][1],read_def=True),
                    }
                    for operand in plat_mem.operands
                ])
                memory_hierarchy_graph.add_memory(
                    memory_instance=mem_inst,
                    operands=rename_operands(plat_mem.operands),
                    port_alloc=port_alloc,
                    served_dimensions="all",
                )

            return memory_hierarchy_graph


        # Non usati per il momento
        def get_operational_array(optimal_spatial_mapping:List[Any]=[]):
            """Multiplier array variables"""
            multiplier_input_precision = [8, 8]
            multiplier_energy = 0.04
            multiplier_area = 1
            dimensions={f"D{idx+1}":opt_spat[1] for idx,opt_spat in enumerate(optimal_spatial_mapping)}
            multiplier = Multiplier(
                multiplier_input_precision, multiplier_energy, multiplier_area
            )
            multiplier_array = MultiplierArray(multiplier, dimensions)

            return multiplier_array


        # Sovrascritto
        def get_dataflows(optimal_spatial_mapping:List[Any]=[]):
            return [{f"D{idx+1}":(opt_spat[0],opt_spat[1]) for idx,opt_spat in enumerate(optimal_spatial_mapping)}]
        
        def get_core(id,optimal_spatial_mapping:List[Any]=[],platform_memories:List[MemoryInst]=[]):
            operational_array = get_operational_array(optimal_spatial_mapping=optimal_spatial_mapping)
            #get the memory hierarchy, from the l2 to the register level
            memory_hierarchy = get_memory_hierarchy(operational_array,platform_memories=platform_memories)
            dataflows = get_dataflows(optimal_spatial_mapping=optimal_spatial_mapping)
            core = Core(id, operational_array, memory_hierarchy, dataflows)
            return core

        def get_accelerator(optimal_spatial_mapping:List[Any]=[],platform_memories:List[MemoryInst]=[]):
            """Generate a ZigZag architecture"""
            cores = {get_core(1,optimal_spatial_mapping=optimal_spatial_mapping,platform_memories=platform_memories)}
            acc_name = 'MATCH'
            return Accelerator(acc_name, cores)
        
        ex_module_acc=self.exec_module.zigzag_architecture(optimal_spatial_mapping=optimal_spatial_mapping,platform_memories=platform_memories,match_node=self.match_node)
        if ex_module_acc is not None:
            return ex_module_acc
        return get_accelerator(optimal_spatial_mapping=optimal_spatial_mapping,platform_memories=platform_memories)
    
    def get_dim_name_by_name(self,name):
        if name=="K":
            return self.o_tensor.dims[1]
        if name=="C":
            return self.o_tensor.dims[1]
        if name=="OY":
            return self.o_tensor.dims[2] if len(self.o_tensor.dims)>2 else self.match_node.default_dim
        if name=="OX":
            return self.o_tensor.dims[3] if len(self.o_tensor.dims)>3 else self.match_node.default_dim
        if name=="FY":
            return self.w_tensor.dims[2] if len(self.w_tensor.dims)>2 else self.match_node.default_dim
        if name=="FX":
            return self.w_tensor.dims[3] if len(self.w_tensor.dims)>3 else self.match_node.default_dim
        if name=="IY":
            return self.i_tensor.dims[2] if len(self.i_tensor.dims)>2 else self.match_node.default_dim
        if name=="IX":
            return self.i_tensor.dims[3] if len(self.i_tensor.dims)>3 else self.match_node.default_dim
        if name=="B":
            return self.o_tensor.dims[0] if len(self.o_tensor.dims)>0 else self.match_node.default_dim
        return self.match_node.default_dim

    def zigzag_set_exec_module(self):
        # self.exec_module.set_match_node(self.match_node)
        # set spatial mapping and other known stuff
        self.i_tensor = [t for t in self.match_node.tensors_arr if t.tensor_type == "var"][0]
        self.x_tensor = self.i_tensor
        self.y_tensor = None if len([t for t in self.match_node.tensors_arr if t.tensor_type == "var"])<2 else [t for t in self.match_node.tensors_arr if t.tensor_type == "var"][1]
        self.o_tensor = [t for t in self.match_node.tensors_arr if t.tensor_type=="output"][0]
        self.w_tensor = None if len([t for t in self.match_node.tensors_arr if t.tensor_type == "const"])==0 else [t for t in self.match_node.tensors_arr if t.tensor_type == "const"][0]
        
        self.zigzag_operands = ["I","W","O"] if any([op in self.match_node.ops_occurrences for op in ["conv2d","dense"]]) else ["X","Y","O"]
        self.zigzag_operands_to_tensors = {
            "I":self.i_tensor,
            "W":self.w_tensor,
            "O":self.o_tensor,
            "X":self.x_tensor,
            "Y":self.y_tensor
        }
        self.exec_module.zigzag_set_optimal_spatial_mapping(match_node=self.match_node,pattern_name = self.pattern_name)
        self.exec_module.match_memories(self.pattern_name,self.zigzag_operands)
        self.platform_memories=self.exec_module.platform_memories
        def op_link_name(operand: str="O"):
            if operand=="O":
                return "O"
            elif operand in ["I","X"]:
                return "I1"
            else:
                return "I2"
        self.spatial_mapping = {
            self.pattern_name:
                {
                    "core_allocation": 1,
                    "spatial_mapping":  {f"D{opt_idx+1}":(opt_sptmap[0],self.exec_module.limit_spatial_mapping_to(\
                        dim_size=self.get_dim_name_by_name(opt_sptmap[0]).size,optimal_spat=self.exec_module.get_optimal_spat_size(opt_sptmap[1],self.get_dim_name_by_name(opt_sptmap[0]))))\
                                        for opt_idx,opt_sptmap in enumerate(self.exec_module.zigzag_optimal_spatial_mapping)},
                    "memory_operand_links": {op:op_link_name(op) for op in self.zigzag_operands},
                    "unordered_loops":["FX","FY","C"],#+(["C"] if "nn.dense" != pattern_inst.ordered_operation else []),
                }
        }
        self.cost_model=self.exec_module.cost_model()
        self.exec_module.match_specific_pattern(match_node=self.match_node,pattern_name=self.pattern_name)
        self.match_node.specific_pattern=self.exec_module.specific_pattern
        self.exec_module.match_layout_operand(pattern_name=self.pattern_name,specific_pattern=self.exec_module.specific_pattern,operands=self.zigzag_operands)
        self.optimal_spatial_mapping=self.exec_module.zigzag_optimal_spatial_mapping

    def generate_schedule(self): 
        self.zigzag_set_exec_module()
        self.accelerator = self.generate_accelerator(platform_memories=self.platform_memories,
                                                     optimal_spatial_mapping=self.optimal_spatial_mapping,)
        self.workload[1]["match_node"] = self.match_node
        try:
            current_spatial_mapping = self.spatial_mapping
            found_valid_temporal_mapping = False
            while not found_valid_temporal_mapping:
                try:
                    print("Looking for temporal mapping with following spatial mapping",current_spatial_mapping)
                    self.energy, self.latency, cme = api.get_hardware_performance_zigzag(
                        workload=self.workload,
                        accelerator=self.accelerator,
                        mapping=current_spatial_mapping,
                        opt="latency",
                        dump_filename_pattern=f"tmp/match-layer_?.json",
                        pickle_filename=f"tmp/match-saved_list_of_cmes.pickle",
                        lpf_limit=self.lpf_limit,
                        cost_model_class= self.cost_model
                    )
                    if hasattr(cme[0][0],"is_tm_valid"):
                        found_valid_temporal_mapping = cme[0][0].is_tm_valid
                except NoValidLoopOrderingFoundException as exc:
                    found_valid_temporal_mapping = False
                if not found_valid_temporal_mapping and all([v[1]==1 for v in list(current_spatial_mapping[self.pattern_name]["spatial_mapping"].values())]):
                    raise NoValidLoopOrderingFoundException(
                        f"No valid loop ordering was found for layer {cme[0][0].layer}."
                    )
                if not found_valid_temporal_mapping:
                    max_spatial_size = max(list(current_spatial_mapping[self.pattern_name]["spatial_mapping"].values()),key= lambda a: a[1])
                    curr_ = current_spatial_mapping[self.pattern_name]["spatial_mapping"]
                    for dim_ in curr_.keys():
                        dim_spat_size = current_spatial_mapping[self.pattern_name]["spatial_mapping"][dim_]
                        if dim_spat_size==max_spatial_size:
                            current_spatial_mapping[self.pattern_name]["spatial_mapping"][dim_] = (max_spatial_size[0],floor(max_spatial_size[1]/2))
                            break

        except Exception as exc:
            #breakpoint()
            self.energy=-1
            self.latency=-1
            self.cme=None
            print(f"[ZIGZAG_ENGINE] No valid loop ordering found: {exc}")
            raise Exception(f"[ZIGZAG_ENGINE] No valid loop ordering found: {exc}")
        self.cme = cme[0][0]
        self.zigzag_temporal_mapping = self.cme.temporal_mapping.mapping_dic_stationary
        #breakpoint()
        if self.debug_node_schedule:
            print(f"[ZIGZAG_ENGINE] Total node energy = {self.energy} pJ")
            print(f"[ZIGZAG_ENGINE] Total node latency = {self.latency} cycles")
            print("[ZIGZAG_ENGINE] ZigZag Schedule: ")
            print_mapping(self.cme)

    def transform_schedule(self):
        mem_op_to_layer_op = self.cme.mem_op_to_layer_op
        mem_name = {}
        for mem_op, mems_all_levels in self.cme.accelerator.cores[0].mem_hierarchy_dict.items():
            layer_op = mem_op_to_layer_op[mem_op]
            mem_name[layer_op] = []
            for mem_a_level in mems_all_levels:
                mem_name[layer_op].append(mem_a_level.name)

        self.temporal_mapping = []
        for layer_op, tm_layer_levels in self.zigzag_temporal_mapping.items():
            layerfound = []
            for idx, levels in enumerate(tm_layer_levels):
                for loop_name, loop_size in levels:
                    nameidx = sum([loop_name == el for el in layerfound])
                    fullname = f"{loop_name}_{nameidx}" if nameidx > 0 else loop_name
                    layerfound.append(loop_name)
                    if fullname not in [el["fullname"] for el in self.temporal_mapping]:
                        self.temporal_mapping.append(
                            {
                                "name": loop_name,
                                "index": nameidx,
                                "fullname": fullname,
                                "size": loop_size,
                                "new_size": loop_size,
                                f"mem_{layer_op}": mem_name[layer_op][idx],
                            }
                        )
                    else:
                        self.temporal_mapping[[el["fullname"] for el in self.temporal_mapping].index(fullname)][
                            f"mem_{layer_op}"
                        ] = mem_name[layer_op][idx]
        # reverse it and add spatial dimensions
        self.temporal_mapping=self.temporal_mapping[::-1]
        #self.match_node.operands
        for (spatial_dim,spatial_val) in self.spatial_mapping[self.pattern_name]["spatial_mapping"].values():
            for idxox in range(len(self.temporal_mapping)):
                    if self.temporal_mapping[idxox]["name"] == spatial_dim:
                        self.temporal_mapping[idxox]["index"] += 1
                        self.temporal_mapping[idxox][
                            "fullname"
                        ] = f'{spatial_dim}_{self.temporal_mapping[idxox]["index"]}'
            obj = {
                "name": spatial_dim,
                "fullname": spatial_dim,
                "size": spatial_val,
                "new_size": spatial_val,
                "index": 0,
            }
            for operand in self.zigzag_operands:
                obj[f"mem_{operand}"] = self.temporal_mapping[len(self.temporal_mapping) - 1][f"mem_{operand}"]
            self.temporal_mapping.append(obj)
        
        # FIX THAT O IS LAST TO BE MOVED
        for idx in reversed(range(len(self.temporal_mapping))):
            for op in set(self.zigzag_operands)-set("O"):
                if self.temporal_mapping[idx][f"mem_{op}"]!=mem_name[op][0] and self.temporal_mapping[idx]["mem_O"]==mem_name["O"][0]:
                    self.temporal_mapping[idx]["mem_O"]=mem_name["O"][1 if len(mem_name["O"])>1 else 0]
                    break
        new_temporal_mapping = []
        dim_step = self.workload[1]["loop_dim_size"]
        for idx,t_map in enumerate(self.temporal_mapping):
            dim_step[t_map["name"]] /= t_map["size"]
            t_map["step"] = dim_step[t_map["name"]]
            if t_map["size"] > 1 or idx==0:
                if idx>0:
                    curr_t_map = {k: v for k, v in t_map.items() if k not in ("fullname","index","step","new_size")}
                    last_t_map = {k: v for k, v in new_temporal_mapping[-1].items() if k not in ("fullname","index","step","new_size")}
                    if curr_t_map == last_t_map:
                        t_map["new_size"] *= new_temporal_mapping[-1]["new_size"]
                        new_temporal_mapping[-1] = t_map
                    else:
                        new_temporal_mapping.append(t_map)
                else:
                    new_temporal_mapping.append(t_map)
        self.temporal_mapping = new_temporal_mapping
        self.top_memories = {op:mem_name[op][-1] for op in self.zigzag_operands}
        mem_hierarchy = self.exec_module.mem_hierarchy
        mem_hierarchy_dict = {mem.name:mem for mem in set([mem_ for k,v in self.exec_module.mem_hierarchy.items() for mem_ in v])}
        self.schedule = MatchSchedule(
            [
                MatchBlock(
                    [
                        MatchLoop(
                            name=tm["fullname"],
                            dim=self.get_dim_name_by_name(tm["name"]),
                            size=tm["new_size"],
                            step=tm["step"],
                            mem_transfers=[
                                MatchMemTransfer(
                                    tensor= self.zigzag_operands_to_tensors[op_type],
                                    top_mem=self.top_memories[op_type] if idx==0 else self.temporal_mapping[idx-1][f"mem_{op_type}"],
                                    mem=tm[f"mem_{op_type}"],
                                    sw_controlled=mem_hierarchy_dict[tm[f"mem_{op_type}"]],
                                )
                                for op_type in self.zigzag_operands
                                if (idx==0 and tm[f"mem_{op_type}"]!=self.top_memories[op_type]) or (idx>0 and tm[f"mem_{op_type}"]!=self.temporal_mapping[idx-1][f"mem_{op_type}"]) 
                            ],
                        ) for idx,tm in enumerate(self.temporal_mapping)
                    ],
                    backend="ZigZag"
                )
            ],
            # ZigZag schedule shouldnt use intermediate tensors
            tensors={tens_name:tens for tens_name,tens in self.match_node.tensors.items() if tens.tensor_type!="intermediate"},
        )
        breakpoint()
        # ZigZag expects all the constants to be loaded immediately to the inner memory of weights...
        for const_tensor in self.match_node.const_tensors.values():
            if const_tensor!=self.w_tensor:
                self.schedule.blocks[0].loops[0].mem_transfers.append(
                    MatchMemTransfer(tensor=const_tensor,
                                     top_mem=mem_hierarchy["const"][-1].name,mem=mem_hierarchy["const"][0].name,
                                     sw_controlled=mem_hierarchy["const"][0].sw_controlled)
                )
        memories = self.exec_module.get_all_memories()
        for tensor in self.schedule.tensors.values():
            self.schedule.tensor_tiles[tensor.name] = [MatchTensorTile(tensor=tensor,
                                            tiled_dims=[MatchTiledDim(dim=dim,size=dim.size) for dim in tensor.dims]) for mem in mem_hierarchy_dict]
            for mem_idx,mem_name in enumerate(memories):
                # is a memory of the tensor and its not the top memory so we can get the tiling size of it
                t_type = "out" if tensor.tensor_type=="output" else "const" if tensor.tensor_type=="constant" else "var"
                if mem_name in mem_hierarchy[t_type] and mem_name!=mem_hierarchy[t_type][-1]:
                    steps = {dim.name:dim.size for dim in tensor.dims}
                    for loop in self.schedule.blocks[0].loops:
                        steps[loop.dim.name] = loop.step
                        if any([mem_trans.tensor==tensor and mem_trans.mem==mem_name for mem_trans in loop.mem_transfers]):
                            for dim_idx,dim in enumerate(tensor.dims):
                                if dim.dim_dependency:
                                    new_size = 0
                                    for ind_dim,mult in dim.dim_dependency.dependencies.items():
                                        new_size = (mult*(ind_dim if not hasattr(ind_dim,"name") else steps[ind_dim.size]))
                                    self.schedule.tensor_tiles[tensor.name][mem_idx][dim_idx] = new_size
                                else:
                                    self.schedule.tensor_tiles[tensor.name][mem_idx][dim_idx] = steps[dim.name]