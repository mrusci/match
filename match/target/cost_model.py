from math import ceil, prod
from typing import Callable

from zigzag.classes.cost_model.cost_model import CostModelEvaluation
from zigzag.classes.mapping.temporal.temporal_mapping import TemporalMapping
from zigzag.classes.opt.temporal.loma.engine import NoValidLoopOrderingFoundException

class ZigZagMatchCostModel(CostModelEvaluation):
    """MATCH implementation of the cost model that will be used by ZigZag
    """
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        temporal_mapping_dict=temporal_mapping.mapping_dic_stationary
        operands_=temporal_mapping.operand_list
        constrained_temporal_mapping_dict,valid=self.adjust_temporal_mapping(temporal_mapping_dict,operands_,layer)
        constrained_temporal_mapping=TemporalMapping(temporal_mapping_dict=constrained_temporal_mapping_dict,
                                                     layer_node=temporal_mapping.layer_node)
        self.is_tm_valid=valid
        super(ZigZagMatchCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=constrained_temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access)

    def is_temporal_mapping_valid(self,temporal_mapping_dict,unordered_loops):
        loops_at_outer_level=[lp[0] for lp in temporal_mapping_dict["O"][1]]
        return sum([lp in loops_at_outer_level for lp in unordered_loops])==0

    def adjust_temporal_mapping(self,temporal_mapping_dict,operand_list,layer):
        """Fix the temporal mapping of a schedule to match the requirements of the platform, the default implementation will
        move loops of the output to permit the computation to happen as soon as the output has been allocated

        Args:
            temporal_mapping_dict (Dict[List[List[Tuple]]]): dictionary containing per each operator the list of memories with the loops assigned
            operand_list (List[Str]): operands used for the specific pattern

        Returns:
            Dict[List[List[Tuple]]]: the new temporal mapping satisfying each constraint
        """
        min_innermost_loops=min([len(temporal_mapping_dict[operand][0]) for operand in operand_list])
        temporal_mapping_dict["O"][1]=temporal_mapping_dict["O"][0][min_innermost_loops:]+temporal_mapping_dict["O"][1]
        temporal_mapping_dict["O"][0]=temporal_mapping_dict["O"][0][:min_innermost_loops]
        return temporal_mapping_dict,self.is_temporal_mapping_valid(temporal_mapping_dict,layer.layer_attrs["unordered_loops"])

    def set_match_params(self):
        self.temp_mapping = self.temporal_mapping.mapping_dic_stationary
        self.loop_sizes = self.layer.loop_dim_size
        self.partial_relevant_loop_sizes = self.layer.pr_loop_dim_size
        self.operands = self.temporal_mapping.operand_list
        self.input_operands = [op for op in self.operands if op!="O"]
        self.operand_loops = self.layer.operand_loop_dim
        self.spatial_sizes = self.spatial_mapping.spatial_loop_dim_size
        self.pattern_name = self.layer.layer_attrs["operator_type"]
        self.layer_data = self.layer.layer_attrs["match_layer_data"]

    def def_innermost_loops_cost(self):
        """This function computes the cost of each single iteration of the kernel

        Returns:
            number: The cost of each iteration of the inner computation
        """
        return prod([self.loop_iters_per_mem_level[operand][0] for operand in self.operands])

    def calc_innermost_loops_cost(self):
        self.innermost_loops_cost_per_it=self.def_innermost_loops_cost()

    def calc_loop_iters_per_mem_level(self):
        self.loop_iters_per_mem_level = {
            operand: [prod([v[1] for v in val[mem_level]]) for mem_level in range(len(val))]
            for (operand, val) in self.temp_mapping.items()
        }
        self.outermost_loop_iters = {
            operand: prod([self.loop_iters_per_mem_level[operand][idx+1] 
                           for idx in range(len(self.loop_iters_per_mem_level[operand])-1)])
            for operand in self.operands
        }
    
    def calc_relevancy_map(self):
        self.relevancy_map = self.layer_data.ordered_relevant_loops

    def calc_sizes_per_mem_level(self):
        self.size_per_mem_level = {
            operand: {
                reldim: [prod(
                    [val[1] for m_lev in range(memory_level+1) for val in self.temp_mapping[operand][m_lev] if val[0] == reldim] +
                    [val[1] for val in self.spatial_sizes if val[0]==reldim] + [
                        1 if operand not in self.input_operands else self.layer_data.strides[0 if reldim=="OY" else 1]
                    ]
                ) for memory_level in range(len(self.temp_mapping[operand]))]
                for reldim in self.relevancy_map[operand]
            }
            for operand in self.operands
        }

    def def_transfer_cost(self):
        """This function computes the cost of an iteration of memory transfer per each operand

        Returns:
            Dict[Str,Number]: Cost of transfer per each iteration for every single operand
        """
        return {
            operand:self.input_transfer_costs[operand][1] if operand in self.input_operands else self.output_transfer_costs[1]
            for operand in self.operands
        }

    def calc_transfer_costs(self):
        self.input_transfer_costs=self.data_loading_cc_pair_combined_per_op
        self.output_transfer_costs=self.data_offloading_cc_pair_combined
        self.transfer_costs=self.def_transfer_cost()

    def overall_latency_sync(self):
        self.total_latency=self.input_overall_transfers+self.self.output_overall_transfers+self.computational_cost
    
    def overall_latency_async(self):
        sorted_multiplicities=sorted(set([self.outermost_loop_iters[operand] for operand in self.operands]))
        cycles=0
        prev_mult_=0
        #print(f"Cost model multiplicities {sorted_multiplicities}")
        for idx,mult_ in enumerate(sorted_multiplicities):
            if idx==0:
                cycles+=max([0]+[self.transfer_costs[operand] for operand in self.operands if operand!='O' and self.outermost_loop_iters[operand]>=mult_])
                prev_mult_=1
            cycles+=(mult_-prev_mult_)*max([self.innermost_loops_cost_per_it,max([self.transfer_costs[operand] for operand in self.operands if self.outermost_loop_iters[operand]>=mult_])])
            prev_mult_=mult_
        self.match_overall_latency=cycles+self.innermost_loops_cost_per_it+self.transfer_costs["O"]

    def def_overall_execution(self):
        self.overall_latency_async()

    def calc_match_overall_latency(self):
        self.set_match_params()
        self.calc_loop_iters_per_mem_level()
        self.calc_relevancy_map()
        self.calc_sizes_per_mem_level()
        self.calc_transfer_costs()
        self.calc_innermost_loops_cost()
        self.def_overall_execution()
    
    def calc_overall_latency(self):
        # use default ZigZag implementation (just to compute some necessary parameters)
        super().calc_overall_latency()
        # call user defined latency function
        self.calc_match_overall_latency()
        # set overall latency
        self.latency_total2=self.match_overall_latency