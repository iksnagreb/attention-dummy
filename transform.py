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
# Precompute constant output nodes
from qonnx.transformation.fold_constants import FoldConstants
# Streamlining transformation: This is a collection of various transformations
from finn.transformation.streamline import Streamline
# Reorder operations
from finn.transformation.streamline.reorder import MoveLinearPastFork
# Convert from QONNX model to FINN operators
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN


# Script entrypoint
if __name__ == '__main__':
    # Load the model graph
    model = ModelWrapper("attention.onnx")

    # Add shape and datatype annotations throughout all the graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Cleanup the graph by removing redundant, unnecessary and constant nodes
    # and tensors and gie unique names to everything remaining
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(FoldConstants())

    # Convert from QONNX graph to FINN nodes/operators
    #   Note: In particular, this converts Quanto nodes to MultiThreshold
    model = model.transform(ConvertQONNXtoFINN())

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

    # Save the transformed graph
    model.save("attention.transformed.onnx")
