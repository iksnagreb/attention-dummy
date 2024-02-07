# Protobuf onnx graph node type
from onnx import NodeProto  # noqa
# Helper for creating ONNX nodes
from onnx import helper as oh  # noqa
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Clean-up transformations
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames
)

# Detect and map scaled dot-product attention to FINN custom operator
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
# Cleanup transformations
from transformation.remove import RemoveIdentityTranspose
from transformation.squeeze import Squeeze


# Script entrypoint
if __name__ == '__main__':
    # Load the model graph
    model = ModelWrapper("attention.transformed.onnx")

    # Try to infer reshaping of attention heads
    model = model.transform(InferMultiHeads())  # noqa: Duplicate
    # Try to mode the mult-head splitting past the multi thresholds
    model = model.transform(MoveSplitMultiHeadsPastMultiThreshold())
    # Try to infer a ScaledDotProductAttention custom op
    #   Note: No further transformations can be run after this currently, as
    #   using a finn custom-op cannot be looked up for shape inference.
    model = model.transform(InferScaledDotProductAttention())
    # Parallelize attention head in the onnx graph
    model = model.transform(UnrollMultiHeadAttention())
    # Swap the order of merging the multi heads and applying thresholds
    model = model.transform(MoveMergeMultiHeadsPastMultiThreshold())
    # If applicable, absorb the final thresholds into the attention operator
    model = model.transform(AbsorbMultiThresholdIntoScaledDotProductAttention())

    # Remove dimensions of size 1 (single batch tensors)
    model = model.transform(Squeeze())
    model = model.transform(RemoveIdentityTranspose())
    # Clean up the names for debugging
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Save the inferred graph
    model.save("attention.inferred.onnx")
