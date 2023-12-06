
proposed_pattern_matcher={
    "conv2d-gap9":{
        "pattern":[
            ("nn.conv2d",{
                "support_conditions":{

                }
            }),
            ("nn.bias_add",{
                "support_conditions":{
                    "type_in":["int32"],
                }
            }),
            ("right_shift",{
                "pattern_conditions":{
                    "args":{
                        "constant":True,
                    }       
                },
                "support_conditions":{
                    "value_not_gt":31,
                    "value_not_lt":0,
                }
            }),
            ("clip",{}),
            ("cast",{
                "pattern_conditions":{
                    "type":"int8",
                }
            })
        ]
    }
}
# Imports
import tvm
import logging
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant

logger = logging.getLogger("Gap9Cluster")


def _requant_pattern(prev_op):
    """Add requant pattern (right_shift -> clip -> cast) to prev_op"""
    right_shift = is_op("right_shift")(prev_op, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip).has_attr({"dtype": "uint8"})
    return cast


def _biasadd_requant_pattern(linear_op):
    """Add pattern bias_add-requant to linear_op"""

    bias_add = is_op("nn.bias_add")(linear_op, wildcard()) | is_op("add")(linear_op, wildcard())
    return _requant_pattern(bias_add)


def conv2d_pattern():
    """Create pattern for conv2D with optional fused relu."""
    #breakpoint()
    conv2d = is_op("nn.conv2d")(
            wildcard(), wildcard()
    )
    return _biasadd_requant_pattern(conv2d)


def fully_connected_pattern():
    """Create pattern for nn.dense with optional fused relu."""

    fc = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    return _biasadd_requant_pattern(fc)


def element_wise_add_pattern():
    """Create pattern for element-wise-add with optional fused relu."""

    cast_a = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    cast_b = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    add = is_op("add")(cast_a, cast_b)
    return _requant_pattern(add)


def _check_requant(pattern):
    """Check if requant pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the op before this sequence if supported
    """
    cast = pattern
    right_shift = cast.args[0].args[0]

    # Check range of shift factor
    shift_factor = right_shift.args[1].data.numpy()
    if shift_factor < 0 or shift_factor > 31:
        logger.warning("shift factor of accelerator operation must be in range [0, 31], but got {shift_factor}. Acceleration for this op is not supported.")
        return None

    right_shift_input = right_shift.args[0]

    return right_shift_input


def _check_biasadd_requant(pattern):
    """Check if bias_add-requant pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the linear op before this sequence if supported
    """

    right_shift_input = _check_requant(pattern)
    if right_shift_input is None:
        return None

    # For now, we don't support linears without bias
    if str(right_shift_input.op.name) not in ["nn.bias_add", "add"]:
        logger.warning("Found conv/dense op without nn.bias_add. Acceleration for this op is not supported.")
        return None

    bias_add = right_shift_input

    # Check bias dtype
    bias_dtype = bias_add.args[1].checked_type.dtype
    if bias_dtype != 'int32':
        logger.warning(f"Expected nn.bias_add parameters to be of type int32, but got {bias_dtype}. Acceleration for this op is not supported.")
        return None

    return bias_add.args[0]


def check_conv2d(pattern):
    """Check if the Conv2D is supported by the soma dory accelerator"""
    #breakpoint()
    conv2d = _check_biasadd_requant(pattern)
    if conv2d is None:
        return False

    num_output_channels = conv2d.args[1].data.shape[0]
    # Don't offload grouped analog convolutions
    if conv2d.args[1].checked_type.dtype == "int2":
        if conv2d.attrs['groups'] != 1:
            return False

    def is_conv2d_attr_value_supported(attrs, name, supported_values):
        attr = attrs[name]

        if isinstance(attr, tvm.ir.container.Array):
            attr = list(attr)

        if attr not in supported_values:
            logger.warning(f"Expected nn.conv2d {name} to be one of {supported_values}, but got {attr}. " +\
                            "Acceleration for this op is not supported.")
            return False

        return True

    def is_filter_and_padding_supported(attrs):
        kernel_size = list(attrs["kernel_size"])
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]
        supported_kernels = [1, 3, 5, 7]
        if (kernel_h not in supported_kernels) or (kernel_w not in supported_kernels):
            logger.warning(f"Expected nn.conv2d kernel width and height to be one of {supported_kernels}, " +\
                           f"but got {kernel_size}. " +\
                            "Acceleration for this op is not supported.")
            return False

        # In topi, padding is [padt, padl, padb, padr]
        padding = list(attrs["padding"])
        # Only support equal left-right and top-bottom padding
        if (padding[0] != padding[2]) or (padding[1] != padding[3]):
            logger.warning(f"Expected equal top and bottom padding, and equal left and right padding," +\
                           f"but got {[padding[0], padding[2]]} and {[padding[1], padding[3]]}, respectively. " +\
                            "Acceleration for this op is not supported.")
            return False

        # Only support output with same output dimension on accelerator
        if (kernel_w - 2*padding[1] != 1) and (kernel_h - 2*padding[0] != 1):
            expected_pad = [(kernel_w - 1) // 2, (kernel_h - 1) // 2]
            logger.warning(f"Accelerator only supports 'SAME' padding. " +\
                           f"Expected nn.conv2d with kernel size {kernel_size} to have padding {expected_pad}, " +\
                           f"but got {padding[:2]}.")
            return False

        return True


    # check conv2d attributes
    if (not is_filter_and_padding_supported(conv2d.attrs)
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', [[1, 1], [2, 2]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', [[1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', [1, num_output_channels])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_layout', ['OIHW'])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'data_layout', ['NCHW'])):

        return False

    #conv2d_input = conv2d.args[0]
    conv2d_weight = conv2d.args[1]

    weights_dtype = conv2d_weight.data.dtype
    if not weights_dtype.startswith('int'):
        logger.warning(f"Expected Conv2D weights to be of integer type, got {weights_dtype}. \
                        Acceleration for this conv2d is not supported")
        return False

    return True


def check_fully_connected(pattern):
    """Check if the fully connected layer is supported by the soma dory accelerator"""

    fc = _check_biasadd_requant(pattern)
    if fc is None:
        return False

    #fc_input = fc.args[0]
    #fc_weight = fc.args[1]

    return True


def check_element_wise_add(pattern):
    """Check if the element-wise-add layer is supported by the soma dory accelerator"""
    add = _check_requant(pattern)
    if add is None:
        return False

    tensor_shape_a = list(add.args[0].checked_type.shape)
    tensor_shape_b = list(add.args[1].checked_type.shape)
    if tensor_shape_a != tensor_shape_b:
        logger.warning(f"Tensor shapes for element-wise-add don't match:"+\
                " Tensor a: {tensor_shape_a}," + \
                " Tensor b: {tensor_shape_b}." + \
                " Acceleration for this element-wise-add is not supported")
        return False

    return True

def partitioning_patterns():
    return [
        {   
            "name":"gap9cluster_conv2d",
            "pattern_matcher":conv2d_pattern,
            "pattern_limitations":check_conv2d,
        },
        {
            "name":"gap9cluster_dense",
            "pattern_matcher":fully_connected_pattern,
            "pattern_limitations":check_fully_connected,
        },
        {
            "name":"gap9cluster_add",
            "pattern_matcher":element_wise_add_pattern,
            "pattern_limitations":check_element_wise_add,
        },
    ]