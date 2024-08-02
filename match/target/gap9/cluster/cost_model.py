from match.target.cost_model import ZigZagMatchCostModel
from math import prod,ceil,floor

class Gap9ClusterCostModel(ZigZagMatchCostModel):
    def __init__(
        self,
        *,
        accelerator,
        layer,
        spatial_mapping,
        temporal_mapping,
        access_same_data_considered_as_no_access=True,
    ):
        super(Gap9ClusterCostModel,self).__init__(
            accelerator=accelerator,layer=layer,spatial_mapping=spatial_mapping,
            temporal_mapping=temporal_mapping,
            access_same_data_considered_as_no_access=access_same_data_considered_as_no_access)
    
    def def_transfer_cost(self):
        def get_stride_2_op(operand):
            if operand in ['I','X','Y']:
                return self.loop_sizes['C' if 'C' in self.size_per_mem_level[operand] else 'K']*self.partial_relevant_loop_sizes['IX']
            elif operand=='W':
                return self.loop_sizes['C']*self.loop_sizes['FY']*self.loop_sizes['FX']
            elif operand=='O':
                return self.loop_sizes['K']*self.loop_sizes['OX']
        def get_stride_1_op(operand):
            if operand in ['I','X','Y']:
                return self.loop_sizes['C' if 'C' in self.size_per_mem_level[operand] else 'K']
            elif operand=='W':
                return self.loop_sizes['C']
            elif operand=='O':
                return self.loop_sizes['K']
        def get_num_2d_copies_op(operand):
            if operand in ['I','X','Y']:
                return self.size_per_mem_level[operand]["OY"][0]
            elif operand=='W':
                return self.size_per_mem_level["W"]["K"][0] if self.pattern_name!='depthwise_conv_2d' else 1
            elif operand=='O':
                return self.size_per_mem_level["O"]["OY"][0]
        def get_num_1d_copies_op(operand):
            if operand in ['I','X','Y']:
                return self.size_per_mem_level[operand]["OX"][0]
            elif operand=='W':
                return self.loop_sizes['FY']*self.loop_sizes['FX']*self.loop_sizes['C'] if self.pattern_name!='depthwise_conv_2d' else 1
            elif operand=='O':
                return self.size_per_mem_level["O"]["OX"][0]
        def get_len_1d_copy_op(operand):
            if operand in ['I','X','Y']:
                return self.size_per_mem_level[operand]['C' if 'C' in self.size_per_mem_level[operand] else 'K'][0]
            elif operand=='W':
                return self.loop_sizes['C'] if self.pattern_name!='depthwise_conv_2d' else (self.size_per_mem_level["W"]["K"][0])*self.loop_sizes['FY']*self.loop_sizes['FX']
            elif operand=='O':
                return self.size_per_mem_level["O"]["K"][0]
        
        dmaconfstruct={
            operand:{
                'hwc_to_cwh':operand=='I' and self.pattern_name=='depthwise_conv_2d',
                'stride_2d':get_stride_2_op(operand),
                'stride_1d':get_stride_1_op(operand),
                'num_2d_copies':get_num_2d_copies_op(operand),
                'num_1d_copies':get_num_1d_copies_op(operand),
                'len_1d_copy':get_len_1d_copy_op(operand),
            } for operand in self.operands
        }
        def calc_overhead(operand):
            if dmaconfstruct[operand]['hwc_to_cwh']:
                return (27*dmaconfstruct[operand]['len_1d_copy'])+1000
            elif dmaconfstruct[operand]['num_2d_copies']==1 and dmaconfstruct[operand]['num_1d_copies']==1:
                return 100+300
            else:
                return (27*dmaconfstruct[operand]['num_2d_copies'])+300
            
        overhead_per_op={operand:calc_overhead(operand) for operand in self.operands}

        def calc_total_transfer_cost_per_op(operand):
            if operand=='O':
                return self.output_transfer_costs[0]+overhead_per_op['O']
            else:
                return (self.input_transfer_costs[operand][0]*(2 if dmaconfstruct[operand]['hwc_to_cwh'] else 1))+overhead_per_op[operand]
        
        self.dmaconfstruct=dmaconfstruct
        self.overhead_per_op=overhead_per_op
        return {operand:calc_total_transfer_cost_per_op(operand) for operand in self.operands}
    
    def def_innermost_loops_cost(self):
        def _floor(ch, N):
            return floor((ch + N - 1) / N)
        latency=0
        ch_in = self.dmaconfstruct[self.input_operands[0]]['len_1d_copy']
        ch_out = self.dmaconfstruct['O']['len_1d_copy']
        kernel_size_x = self.loop_sizes['FX']
        kernel_size_y = self.loop_sizes['FY']
        output_shape=[1,self.dmaconfstruct['O']['len_1d_copy'],self.dmaconfstruct['O']['num_2d_copies'],self.dmaconfstruct['O']['num_1d_copies']]
        strides=self.layer_data.strides
        if self.layer_data.specific_pattern in ["conv2d","pointwise_conv2d"]:
            iterations = _floor(int(ch_out), 4) * _floor(int(output_shape[2]*strides[0]), 8)* _floor(int(output_shape[3]*strides[1]), 2)
            im2col = kernel_size_x * kernel_size_y * ch_in * 2 // 4
            matmul = (5 + _floor(kernel_size_x * kernel_size_y * ch_in, 4) * (6 + 8) + 10)
            latency = iterations * (im2col + matmul)
        elif self.layer_data.specific_pattern in ['depthwise_conv2d','depthwise_conv2d_less_4']:
            # 1 MAC/cycle
            latency = 4 * _floor(ch_out, 8) * _floor(output_shape[3]*strides[1],4) * kernel_size_x * kernel_size_y * int(output_shape[2]*strides[0])
        elif self.layer_data.specific_pattern=='dense':
            latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        elif self.layer_data.specific_pattern in ['conv2d_sparse_4', 'conv2d_sparse_8', 'conv2d_sparse_16']:
            OX = output_shape[3]; OY = output_shape[2]
            # TODO : Inserire qua N e M a seconda dello sparsity pattern
            N = 1; SOFWARE_OP = False
            if self.layer_data.specific_pattern == 'conv2d_sparse_4':
                M = 4
            elif self.layer_data.specific_pattern == 'conv2d_sparse_8':
                M = 8
            elif self.layer_data.specific_pattern == 'conv2d_sparse_16':
                M = 16
            # K is now 1. Kernels are 1x2 due to N:M sparsity being on the C*FX*FY dimension
            # (No indices are shared over K)
            iterations =  _floor(int(ch_out), 1) * _floor(int(OY * strides[0]), 8)* _floor(int(OX * strides[1]), 2)
            # Dense im2col
            im2col = (kernel_size_x * kernel_size_y * ch_in * 2  // 4 if (kernel_size_x * kernel_size_y) != 1 else 0)
            # Sparse Matmul - Software
            sparse_dim = ((kernel_size_x * kernel_size_y * ch_in) * N) // M
            if SOFWARE_OP:
                if M == 4:
                    n_op_getmask = 1 + 2 + 2 + 1; add_shift = 1 + 1 + 1; loads = 1 + 4 + 4 + 1; ptr_update = 2; macs = 2
                    matmul_cost = n_op_getmask + add_shift + loads + ptr_update + macs
                elif M in [8, 16]:
                    n_op_getmask = 2 + 2; add_shift = 1 + 1 + 1; loads = 2 + 4 + 4 + 1; ptr_update = 2; macs = 2
                    matmul_cost = n_op_getmask + add_shift + loads + ptr_update + macs
                matmul = _floor(sparse_dim, 4) * (matmul_cost) + 10
                latency = iterations * (im2col + matmul)
            else:  # Extended ISA
                n_op_getmask = 0
                if M == 4:
                    n_custom_op = 16; loads = 1 + 2; macs = 4
                elif M in [8, 16]:
                    n_custom_op = 8; loads = 2; macs = 2
                matmul_cost = n_op_getmask + n_custom_op + loads + macs
                matmul = _floor(sparse_dim, 4) * (matmul_cost) + 10 + 5
                latency = iterations * (im2col + matmul)
        elif self.layer_data.specific_pattern == "dense_sparse":
            N_CORES = 8; SOFWARE_OP = True; N = 1; M = 4
            # Kernel is x2, unrolled on the output dim.
            sparse_dim = (ch_in * N) // M
            im2col = 0
            if SOFWARE_OP:
                iterations = _floor(int(ch_out), N_CORES)
                if M == 4:
                    n_op_getmask = 1 + 2 + 2 + 1; add_shift = 1 + 1 + 1; loads = 1 + 4 + 1; ptr_update = 1; macs = 1
                    matmul_cost = n_op_getmask + add_shift + loads + ptr_update + macs
                else:
                    n_op_getmask = 1 + 1 + 1 + 1; add_shift = 1 + 1 + 1; loads = 2 + 4 + 1; ptr_update = 1; macs = 1
                    matmul_cost = n_op_getmask + add_shift + loads + ptr_update + macs
                matmul = _floor(sparse_dim, 4) * (matmul_cost)
                latency += iterations * (matmul)
            else:
                iterations = _floor(int(ch_out), N_CORES * 2)
                n_op_getmask = 0
                if M == 4:
                    loads = 4 + 1; n_custom_op = 16; macs = 4; matmul_cost = loads + n_custom_op + macs
                else:
                    loads = 4 + 1; n_custom_op = 16; macs = 2; matmul_cost = loads + n_custom_op + macs
                matmul = _floor(sparse_dim, 4) * (matmul_cost)
                latency += iterations * (matmul)
        else:
            latency += _floor(ch_in, 2) * _floor(ch_out, 4)
        return latency
    

    # SPARSE START


    # # SPARSE END