# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX graph transformations for renaming and cleaning up
from qonnx.transformation.general import (
    GiveUniqueNodeNames,
    GiveReadableTensorNames,
    RemoveUnusedTensors,
    RemoveStaticGraphInputs,
    ConvertDivToMul
)
# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
# Precompute constant output nodes
from qonnx.transformation.fold_constants import FoldConstants
# Streamlining transformation: This is a collection of various transformations
from finn.transformation.streamline import Streamline
# Reorders operations
from finn.transformation.streamline.reorder import MoveScalarMulPastMatMul
# Absorbs operations / parameters into other operations
from finn.transformation.streamline.absorb import AbsorbMulIntoMultiThreshold
# Convert from QONNX model to FINN operators
from finn.transformation.qonnx.convert_qonnx_to_finn import (
    FoldTransposeIntoQuantInit,
    ConvertQuantActToMultiThreshold,
    ConvertQONNXtoFINN
)


# Script entrypoint
if __name__ == '__main__':
    # Load the model graph
    model = ModelWrapper("attention.onnx")

    # Shape and datatype inference transformations
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Some renaming and cleanup transformations
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())

    # Fold constant-output nodes
    model = model.transform(FoldConstants())
    # Convert divisions to multiplications (applies to the scale of the QK
    # matmul output)
    model = model.transform(ConvertDivToMul())

    # # Get rid of the transpose operation applied to the key matrix
    # #   Note: This might be the reason for the shape inference error of the
    # #   ConvertQONNXtoFINN transformation below.
    # model = model.transform(FoldTransposeIntoQuantInit())

    # # Convert from QONNX graph to FINN nodes/operators
    # #   Note: Somehow fails due to shape inference?
    # model = model.transform(ConvertQONNXtoFINN())

    # Turn all quantization layers into MultiThresholds
    model = model.transform(ConvertQuantActToMultiThreshold())
    # Try to apply streamlining transformation
    model = model.transform(Streamline())

    # Save the transformed graph
    model.save("attention.transformed.onnx")
