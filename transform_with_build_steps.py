# QONNX wrapper of ONNX model graphs
from finn.builder.build_dataflow_steps import step_convert_to_hls, step_tidy_up
from qonnx.core.modelwrapper import ModelWrapper
# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveUniqueParameterTensors
)
# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
# Precompute constant output nodes
from qonnx.transformation.fold_constants import FoldConstants
# Streamlining transformation: This is a collection of various transformations
from finn.transformation.streamline import Streamline
# Reorder operations
from finn.transformation.streamline.reorder import (
    MoveLinearPastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarLinearPastInvariants
)
# Convert from QONNX model to FINN operators
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

# Remove some operations without real effect
from transformation.remove import RemoveIdentityTranspose, RemoveIdentityReshape

from build_steps import (
    step_convert_attention_to_hls,  
    step_streamline_norms, 
    step_tidy_up_pre_attention, 
    step_streamline_attention, 
    step_streamline_residual, 
    step_streamline_positional, 
    step_tidy_up_post_attention,
    step_restore_batchnorm_transpose,
)

from transformation.transpose import (
    MoveTransposePastEltwise,
    CollapseRepeatedTranspose,
    RestoreTransposeAfterBatchNorm,
    CombineParallelTransposeAfterBatchNorm
)

import sys
import re

# Script entrypoint
if __name__ == '__main__':
    in_file = sys.argv[1] 
    # assume file ends with .onnx
    out_file_prefix = in_file[:-5]
    file_ending = ".onnx"

    # Load the model graph
    model = ModelWrapper(in_file)
    model = step_restore_batchnorm_transpose(model)
    #model = model.transform(RestoreTransposeAfterBatchNorm())
    #model = model.transform(CombineParallelTransposeAfterBatchNorm())

    model = step_tidy_up(model)
    model = step_tidy_up_pre_attention(model)
    # Convert from QONNX graph to FINN nodes/operators
    #   Note: In particular, this converts Quanto nodes to MultiThreshold
    model = model.transform(ConvertQONNXtoFINN())
    model = step_streamline_positional(model)
    model = step_streamline_attention(model)
    model = step_streamline_residual(model)
    model = model.transform(Streamline())
    model = step_streamline_norms(model)
    model = model.transform(Streamline())
    model = step_convert_attention_to_hls(model)
    model = step_tidy_up_post_attention(model)
    # model = step_convert_to_hls(model)
    
    model = model.transform(Streamline())
    # Save the transformed graph
    model.save(out_file_prefix + ".finn" + file_ending)
