from math import floor
from typing import Any, Dict, List

import numpy as np
from match.codegen.temporal_mapping_engine.temporal_mapping_engine import TemporalMappingEngine
from match.codegen.layer_data import LayerData
from match.hwmodel.hwmodel import HwModel
from match.hwmodel.memory_inst import MemoryInst, PortConnection
from zigzag import api
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.operational_unit import Multiplier
from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core

class ZigZagEngine(TemporalMappingEngine):
    def __init__(self,workload:Dict[str,Any]={},hwmodel:HwModel=None,pattern_name:str="",layer_data:LayerData=None):
        super(ZigZagEngine, self).__init__(workload=workload,hwmodel=hwmodel,pattern_name=pattern_name,layer_data=layer_data)
        self.lpf_limit=13
        self.debuglayer=True
        self.zigzag_temporal_mapping=dict()

    def transform_workload_for_engine(self):
        self.workload={1:self.workload}

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
                    size = floor(plat_mem.k_bytes * 1024 * 8) ,  # Size of L2 memory
                    r_bw=floor(plat_mem.r_bw),
                    w_bw=floor(plat_mem.w_bw),
                    r_cost=100,
                    w_cost=110,
                    area=0,
                    r_port=floor(plat_mem.r_ports),
                    w_port=floor(plat_mem.w_ports),
                    rw_port=floor(plat_mem.rw_ports),
                    latency=1, # TODO: Non usato dentro Zigzag, dovrebbe essere > 1
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
        
        return get_accelerator(optimal_spatial_mapping=optimal_spatial_mapping,platform_memories=platform_memories)
    
    def generate_temporal_mapping(self,spatial_mapping:Dict={},platform_memories:Dict={},optimal_spatial_mapping:List=[],cost_model:Any=None): 
        self.accelerator = self.generate_accelerator(platform_memories=platform_memories,
                                                     optimal_spatial_mapping=optimal_spatial_mapping,)
        self.workload[1]["cost_model"] = cost_model
        self.workload[1]["attrs"] = self.layer_data.layer_attrs
        self.spatial_mapping = spatial_mapping
        energy, latency, cme = api.get_hardware_performance_zigzag(
            workload=self.workload,
            accelerator=self.accelerator,
            mapping=spatial_mapping,
            opt="latency",
            dump_filename_pattern=f"tmp/match-layer_?.json",
            pickle_filename=f"tmp/match-saved_list_of_cmes.pickle",
            lpf_limit=self.lpf_limit
        )
        self.cme = cme[0][0]
        self.zigzag_temporal_mapping = self.cme.temporal_mapping.mapping_dic_stationary
        if self.debuglayer:
            print(f"\n\nOur result Latency was Comp {self.cme.latency_total0} total {self.cme.latency_total2}\n\n")
            self.energy = energy
            self.latency = latency
            print(f"Total network energy = {energy:.2e} pJ")
            print(f"Total network latency = {latency:.2e} cycles")
            print("Mapping")
            print_mapping(self.cme)

    def transform_temporal_mapping(self):
        mem_op_to_layer_op = self.cme.mem_op_to_layer_op
        mem_name = {}
        for mem_op, mems_all_levels in self.cme.accelerator.cores[0].mem_hierarchy_dict.items():
            layer_op = mem_op_to_layer_op[mem_op]
            mem_name[layer_op] = []
            for mem_a_level in mems_all_levels:
                mem_name[layer_op].append(mem_a_level.name)

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
                                f"mem_{layer_op}": mem_name[layer_op][idx],
                            }
                        )
                    else:
                        self.temporal_mapping[[el["fullname"] for el in self.temporal_mapping].index(fullname)][
                            f"mem_{layer_op}"
                        ] = mem_name[layer_op][idx]
        # reverse it and add spatial dimensions
        self.temporal_mapping=self.temporal_mapping[::-1]
        for (spatial_dim,spatial_val) in self.spatial_mapping[self.layer_data.workload_name]["spatial_mapping"].values():
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
                "index": 0,
            }
            for operand in self.layer_data.operands:
                obj[f"mem_{operand}"] = self.temporal_mapping[len(self.temporal_mapping) - 1][f"mem_{operand}"]
            self.temporal_mapping.append(obj)