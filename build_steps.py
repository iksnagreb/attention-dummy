# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Converts ONNX graph nodes to QONNX custom-ops if possible
from qonnx.custom_op.registry import getCustomOp
# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    GiveUniqueParameterTensors,
    ConvertDivToMul
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
    MoveMulPastFork,
    MoveLinearPastFork,
    MoveTransposePastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarLinearPastInvariants,
    MoveTransposePastEltwise,
)
# Collapse consecutive operations of the same type
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedTranspose
)
# FINN transformation converting ONNX nodes to hardware custom operators
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferElementwiseBinaryOperation
)
# Remove some operations without real effect
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape
)
# Cleanup transformation getting rid of 3d data layout
from finn.transformation.squeeze import Squeeze
# Detects the attention pattern and converts to hardware custom op
from finn.transformation.fpgadataflow.attention import (
    InferScaledDotProductAttention,
    AbsorbMultiThresholdIntoScaledDotProductAttention
)
# Mult-Head Attention support
from finn.transformation.fpgadataflow.attention_heads import (
    InferMultiHeads,
    MoveSplitMultiHeadsPastMultiThreshold,
    UnrollMultiHeadAttention,
    MoveMergeMultiHeadsPastMultiThreshold
)
# Stream replication for outputs with multiple consumers
from finn.transformation.fpgadataflow.replicate_stream import (
    InferReplicateStream
)
# FINN dataflow builder configuration
from finn.builder.build_dataflow_config import (
    VerificationStepType, DataflowBuildConfig
)
# Graph transformation setting the folding, i.e., parallelization configuration
from finn.transformation.fpgadataflow.set_folding import SetFolding
# FINN verification after build/graph transformation steps
from finn.builder.build_dataflow_steps import verify_step


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
def step_streamline_attention(model: ModelWrapper, cfg: DataflowBuildConfig):
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

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_attention_python", need_parent=False
        )

    # Return the streamlined model
    return model


# Streamlining transformations to be applied to residual branches
def step_streamline_residual(model: ModelWrapper, cfg: DataflowBuildConfig):
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

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_residual_python", need_parent=False
        )

    # Return the streamlined model
    return model


# Streamlining transformation to be applied to the normalization layers
def step_streamline_norms(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Streamline transposed batch normalization (move transposes past the
    # scale-bias operator, so they can be collapsed afterward)
    model = model.transform(MoveTransposePastEltwise())
    # There should now be transposes next to each other which can be collapsed
    model = model.transform(CollapseRepeatedTranspose())
    # The transposes around the batch normalization should be collapsed by now
    # and cancel each other out
    model = model.transform(RemoveIdentityTranspose())
    # We now might have transpose operations accumulating in front of fork nodes
    model = model.transform(MoveTransposePastFork())
    model = model.transform(MoveTransposePastEltwise())
    model = model.transform(CollapseRepeatedTranspose())
    model = model.transform(RemoveIdentityTranspose())
    # This needs to be done twice, as per block there is one fork to the
    # residual branch and one fork into the queries, keys and values input.
    model = model.transform(MoveTransposePastFork())
    model = model.transform(MoveTransposePastEltwise())
    model = model.transform(CollapseRepeatedTranspose())
    model = model.transform(RemoveIdentityTranspose())
    # This might have caused the normalization scale and bias to accumulate in
    # front of transpose or fork node
    model = model.transform(MoveLinearPastEltwiseAdd())  # noqa: Duplicate
    model = model.transform(MoveLinearPastFork())
    model = model.transform(MoveScalarLinearPastInvariants())
    # This might have enabled more streamlining transformations
    model = model.transform(Streamline())
    # We need a custom streamlining step to enable streamlining through certain
    # fork-nodes Note: This transform is part of finn, but not included in the
    # standard streamlining transformations
    model = model.transform(MoveLinearPastFork())
    # This might have enabled more streamlining transformations
    model = model.transform(Streamline())

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(model, cfg, "streamlined_norms_python", need_parent=False)

    # Return the streamlined model
    return model


# Streamlining transformation to be applied to the positional encoding layer
def step_streamline_positional(model: ModelWrapper, cfg: DataflowBuildConfig):
    # There is probably a division in front of the quantized positional
    # encoding, which is exactly the inverse of the multiplication in front of
    # that: The are the matching scale factors of the shared input quantizer of
    # input and positional encoding. Convert the division to multiplication, so
    # these two can be merged.
    model = model.transform(ConvertDivToMul())
    # Merge the quantization scales of shared input quantizers
    model = model.transform(CollapseRepeatedMul())
    # Push scalar multiplications, probably scale factors of quantizers, into
    # the branches of a fork
    model = model.transform(MoveMulPastFork())

    # If configured, run a verification of the transformed model on some sample
    # inputs
    if (VerificationStepType.STREAMLINED_PYTHON in
            cfg._resolve_verification_steps()):  # noqa
        verify_step(
            model, cfg, "streamlined_positional_python", need_parent=False
        )

    # Return the streamlined model
    return model


# Function running the InferScaledDotProductAttention transformation
def step_convert_attention_to_hw(model: ModelWrapper, _):
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
    # Return the model with attention and multi-heads mapped to hardware
    # operators
    return model


# Function running the transformations to convert elementwise binary operations
# to their hardware implementations
def step_convert_elementwise_binary_to_hw(model: ModelWrapper, _):
    # Convert elementwise operations to hardware operators
    #   Note: Do not convert the final Mul operator at the output
    return model.transform(InferElementwiseBinaryOperation(
        InferElementwiseBinaryOperation.reject_output_dequant
    ))


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
    # If applicable, absorb the final thresholds into the attention operator
    #   Note: Might be applicable again after squeezing a transpose away
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())

    # Squeezing might enable some more streamlining transformations once again
    model = model.transform(MoveLinearPastEltwiseAdd())  # noqa: Duplicate
    model = model.transform(MoveLinearPastFork())
    model = model.transform(MoveScalarLinearPastInvariants())
    # Do the normal streamlining flow once again
    model = model.transform(Streamline())

    # Clean up the names for debugging
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # Return the tidied up model
    return model


# Custom step for setting the parallelism to meet the target of T^2 cycles per
# sequence
def step_set_target_parallelization(seq_len: int,
                                    emb_dim: int):  # noqa: emb_dim
    # The wrapping function is a generator and this is the actual build step
    # function taking the model and build configuration
    def _step_set_target_parallelization(
            model: ModelWrapper, cfg: DataflowBuildConfig
    ):
        # Run over all nodes in the model graph to look for attention operators,
        # which are currently not handled by the SetFolding transformation
        for index, node in enumerate(model.graph.node):
            # Only handle attention operations here
            if node.op_type == "ScaledDotProductAttention_hls":
                # Convert this to the custom-op instance for easy access to node
                # attributes
                inst = getCustomOp(node)
                # Set the sequence and embedding dimension folding to meet the
                # T^2 cycles target, i.e., fully parallel along the embedding
                # dimension and fully sequential along the sequence dimension
                inst.set_nodeattr("EmbFold", 1)
                inst.set_nodeattr("SeqFold", seq_len)
        # Apply the built-in folding configuration transformation with the
        # T^2 target cycles
        model = model.transform(SetFolding(
            seq_len ** 2, cfg.mvau_wwidth_max, cfg.folding_two_pass_relaxation
        ))
        # TODO: Extract the folding configuration
        # Return the model with configured parallelization
        return model

    # Return the wrapped build step function
    return _step_set_target_parallelization


# Custom build step trying to infer appropriate FIFO sizes for attention-related
# operators
def step_infer_fifo_depths(seq_len: int, emb_dim: int):  # noqa: emb_dim
    # The wrapping function is a generator and this is the actual build step
    # function taking the model and build configuration
    def _step_infer_fifo_depths(model: ModelWrapper, _: DataflowBuildConfig):
        # Run over all nodes in the model graph
        for index, node in enumerate(model.graph.node):
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)
            # Extract the FIFO depths configuration of the node
            in_depths = inst.get_nodeattr("inFIFODepths")
            out_depths = inst.get_nodeattr("outFIFODepths")

            # Number of inputs and outputs to/from the node
            num_inputs = len(node.input)
            num_outputs = len(node.output)

            # If the input/output has only default configurations, fill with as
            # many shallow FIFOs as there are inputs, to avoid later problems
            # with to few FIFO depths specified
            if in_depths == [2] and num_inputs > 1:
                in_depths = num_inputs * [2]
            if out_depths == [2] and num_outputs > 1:
                out_depths = num_outputs * [2]

            # Special case: Attention needs properly sized input FIFOs
            if node.op_type == "ScaledDotProductAttention_hls":
                # Each folded input stream needs to be buffered completely
                # TODO: Not exactly sure whether this is always correct or just
                #  the worst-case
                in_depths = [
                    inst.get_number_input_values(i) for i in range(num_inputs)
                ]
                # Note: No special treatment of the output FIFO
                # out_depths = ...

            # Special case: Adding residual branches needs to buffer the inputs
            # to avoid deadlocks if one branch is running faster/slower
            if node.op_type == "ElementwiseAdd_hls":
                # Only relevant if for join-node operations, i.e., node actually
                # consumes two branches, potentially operating at a different
                # rate
                if model.is_join_node(node):
                    # Set both inputs to buffer as many cycles as we target for
                    # the attention operations, i.e., the T^2 cycles per
                    # sequence target
                    # TODO: Not exactly sure whether this is always correct or
                    #  just the worst-case
                    in_depths = [seq_len ** 2, seq_len ** 2]
                    # Note: No special treatment of the output FIFO
                    # out_depths = ...

            # Set the updated FIFO depths attributes
            inst.set_nodeattr("inFIFODepths", in_depths)
            inst.set_nodeattr("outFIFODepths", out_depths)
        # Return the model with configured parallelization
        return model

    # Return the wrapped build step function
    return _step_infer_fifo_depths
