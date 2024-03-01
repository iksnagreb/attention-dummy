# QONNX wrapper of ONNX model graphs
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
# Fuse/Absorb operations
from finn.transformation.streamline.absorb import AbsorbAddIntoMultiThreshold
# Reorder operations
from finn.transformation.streamline.reorder import (
    MoveLinearPastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarLinearPastInvariants
)
# FINN transformation converting ONNX nodes to HLS custom operators
from finn.transformation.fpgadataflow.convert_to_hls_layers import (
    InferAddStreamsLayer
)
# Remove some operations without real effect
from transformation.remove import RemoveIdentityTranspose, RemoveIdentityReshape
# Cleanup transformations
from transformation.squeeze import Squeeze
# Transformations involving Transpose operators
from transformation.transpose import (
    MoveTransposePastEltwise,
    CollapseRepeatedTranspose
)
# Detects the attention pattern and converts to HLS custom op
from transformation.attention import (
    InferScaledDotProductAttention,
    AbsorbMultiThresholdIntoScaledDotProductAttention
)
# Mult-Head Attention support
from transformation.attention_heads import (
    InferMultiHeads,
    MoveSplitMultiHeadsPastMultiThreshold,
    UnrollMultiHeadAttention,
    MoveMergeMultiHeadsPastMultiThreshold
)
# Stream replication for outputs with multiple consumers
from transformation.replicate_stream import InferReplicateStream


# Function running transformations necessary to clean up models containing
# attention operators
def step_tidy_up_pre_attention(model: ModelWrapper, _):
    # Add shape and datatype annotations throughout all the graph
    model = model.transform(InferDataTypes())  # noqa Duplicate
    model = model.transform(InferShapes())

    # Cleanup the graph by removing redundant, unnecessary and constant nodes
    # and tensors and give unique names to everything remaining
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(FoldConstants())

    # Remove unnecessary shape and layout transformations
    model = model.transform(RemoveIdentityReshape())
    model = model.transform(RemoveIdentityTranspose())
    # Insert tensor layout annotations for Quant to MultiThreshold transform
    # to determine the correct output channel dimension
    model = model.transform(InferDataLayouts())
    # Return the tidied up model
    return model


# Variant of streamlining transformations adapted to attention operators
def step_streamline_attention(model: ModelWrapper, _):
    # Apply the set of standard streamlining transformations from finn to the
    # model
    model = model.transform(Streamline())
    # We need a custom streamlining step to enable streamlining through certain
    # fork-nodes Note: This transform is part of finn, but not included in the
    # standard streamlining transformations
    model = model.transform(MoveLinearPastFork())
    # Streamline again there should be more transformations enabled after moving
    # some nodes past forks
    model = model.transform(Streamline())
    # Streamline again there should be more transformations enabled after moving
    # some nodes past forks
    model = model.transform(Streamline())
    # Streamline again there should be more transformations enabled after moving
    # some nodes past forks
    model = model.transform(Streamline())
    # Return the streamlined model
    return model


# Streamlining transformations to be applied to residual branches
def step_streamline_residual(model: ModelWrapper, _):
    # Streamline the residual connections by moving scale factors past
    # elementwise add nodes
    model = model.transform(MoveLinearPastEltwiseAdd())  # noqa: Duplicate
    model = model.transform(MoveLinearPastFork())
    model = model.transform(MoveScalarLinearPastInvariants())
    # Do the normal streamlining flow once again
    model = model.transform(Streamline())
    # Streamline the residual connections by moving scale factors past
    # elementwise add nodes again
    #   TODO: We probably need one round of these streamlining transformations
    #    per transformer block...
    model = model.transform(MoveLinearPastEltwiseAdd())
    model = model.transform(MoveLinearPastFork())
    model = model.transform(MoveScalarLinearPastInvariants())
    # And again to get the last floating-point Mul absorbed into thresholds
    model = model.transform(Streamline())
    # Return the streamlined model
    return model


# Streamlining transformation to be applied to the normalization layers
def step_streamline_norms(model: ModelWrapper, _):
    # Streamline transposed batch normalization (move transposes past the
    # scale-bias operator, so they can be collapsed afterward)
    model = model.transform(MoveTransposePastEltwise())
    # There should now be transposes next to each other which can be collapsed
    model = model.transform(CollapseRepeatedTranspose())
    # The transposes around the batch normalization should be collapsed by now
    # and cancel each other out
    model = model.transform(RemoveIdentityTranspose())
    # This might have enabled more streamlining transformations
    model = model.transform(Streamline())
    # Return the streamlined model
    return model


# Function running the InferScaledDotProductAttention transformation
def step_convert_attention_to_hls(model: ModelWrapper, _):
    # Try to infer reshaping of attention heads
    model = model.transform(InferMultiHeads())  # noqa: Duplicate
    # Try to mode the mult-head splitting past the multi thresholds
    model = model.transform(MoveSplitMultiHeadsPastMultiThreshold())
    # Moving multi-head splitting past multi thresholds might enable absorbing
    # adds into thresholds once again
    model = model.transform(AbsorbAddIntoMultiThreshold())
    # Try to infer a ScaledDotProductAttention custom op
    model = model.transform(InferScaledDotProductAttention())
    # Parallelize attention head in the onnx graph
    model = model.transform(UnrollMultiHeadAttention())
    # Swap the order of merging the multi heads and applying thresholds
    model = model.transform(MoveMergeMultiHeadsPastMultiThreshold())
    # If applicable, absorb the final thresholds into the attention operator
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())
    # Return the model with attention and multi-heads mapped to hls operators
    return model


# Function running the transformations to convert residual branches to HLS
# layers, in particular     model = model.transform(InferAddStreamsLayer())
def step_convert_residual_to_hls(model: ModelWrapper, _):
    # Convert elementwise add operations to streamed adding
    return model.transform(InferAddStreamsLayer())


# Function running the InferReplicateStream transformation
def step_replicate_streams(model: ModelWrapper, _):
    # Properly replicate the stream feeding the query, key and value projections
    return model.transform(InferReplicateStream())


# Post-processing tidy-up squeezing dimensions and identity operators left over
# from mapping the attention operators
def step_tidy_up_post_attention(model: ModelWrapper, _):
    # Remove dimensions of size 1 (single batch tensors)
    model = model.transform(Squeeze())
    model = model.transform(RemoveIdentityTranspose())
    # Squeezing might enable absorbing adds into thresholds once again
    model = model.transform(AbsorbAddIntoMultiThreshold())
    # Clean up the names for debugging
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # Return the tidied up model
    return model
