
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Operations to support the SOMA accelerator.
"""
from match.transform.save import MatchSaveModule, MatchSaveRelay
from match.utils.utils import get_model_name
import tvm
import logging

from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
#from tvm.driver.tvmc import TVMCException

# don't remove this import even if it does not seem to be used
# because this is the point where the match backend is registered
import tvm.relay.backend.contrib.match

from match.target import get_target,MatchTarget,DefaultMatchTarget


logger = logging.getLogger("match")

def pattern_table(target:MatchTarget=DefaultMatchTarget()):
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """
    patterns=[(
        f"match.{target_pattern.name}", 
        target_pattern.pattern(), 
        target_pattern.additional_checks) 
        for target_pattern in target.partitioning_patterns()]
    return patterns

def partition(mod, params, dpu, opts):
    """
    The partitioning sequence for the MATCH byoc
    Parameters
    ----------
    mod The module to use

    Returns
    -------
    The partitioned module.

    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    
    target=get_target()

    pipeline = []
    pipeline.append(MatchSaveRelay("start"))
    pipeline.append(transform.InferType())

    for net_transform_name, net_transform in target.network_transformations(opts):
        pipeline.append(net_transform)
        pipeline.append(MatchSaveRelay(net_transform_name))
        pipeline.append(transform.InferType())

    pipeline.append(MatchSaveRelay("transformed"))
    pipeline.append(transform.MergeComposite(pattern_table(target=target)))
    pipeline.append(transform.AnnotateTarget(["match"]))
    pipeline.append(MatchSaveRelay("merged"))

    for net_transform_name, net_transform in target.adjust_network(opts):
        pipeline.append(net_transform)
        pipeline.append(MatchSaveRelay(net_transform_name))
        pipeline.append(transform.InferType())

    pipeline.append(MatchSaveRelay("adjusted"))
    pipeline.append(transform.PartitionGraph(get_model_name()))
    pipeline.append(transform.InferType())
    pipeline.append(MatchSaveRelay("partitioned"))
    
    pipeline.append(MatchSaveModule())
    seq = tvm.transform.Sequential(pipeline)
    with tvm.transform.PassContext(opt_level=3):
        try:
            fused = seq(mod)
            return fused
        except Exception as err:
            breakpoint()
            raise Exception(
                "Error converting layout to {0}".format(str(err))
            )
