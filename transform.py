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
# Reorder operations
from finn.transformation.streamline.reorder import MoveLinearPastFork
# Convert from QONNX model to FINN operators
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

# Gets items from protobuf by name
from qonnx.util.basic import get_by_name, remove_by_name
# Helper for creating ONNX nodes
from onnx import helper as oh  # noqa
# QONNX graph transformation base class
from qonnx.transformation.base import Transformation
# For array handling
import numpy as np


# Squeezes, i.e., removes, dimensions of size 1
class Squeeze(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # There should not be any squeeze or unsqueeze operations in the
            # graph as these would interfere with this transformation
            assert node.op_type not in {"Squeeze", "Unsqueeze"}, \
                f"Squeezing graph containing {node.op_type}"

            # Validate slice not slicing along squeezed dimension
            if node.op_type == "Slice":
                # Axes to slice along is supplied as the 4th input to the node
                axes = model.get_initializer(node.input[3])
                # If this is an initializer, there are constant axes to slice
                if axes is not None:
                    # Get the shape of the input, assuming the input from
                    # upstream to be the 1st input
                    shape = model.get_tensor_shape(node.input[0])
                    # Slice might operate on multiple axes
                    for axis in axes:
                        # Axis must not refer to a dimension of size 1
                        assert shape[axis] > 1, \
                            f"Slice along dimension to be squeezed: {node.name}"

            # Need to adapt reshape operations to drop dimensions of size 1
            if node.op_type == "Reshape":
                # Second input to the reshape operation is the target shape
                shape = model.get_initializer(node.input[1])
                # If the initializer is present, this is a constant shape
                # reshape which can be replaced by the squeezed shape
                if shape is not None:
                    # Squeeze the shape by removing all dimensions with size 1
                    new_shape = np.asarray([
                        size for size in shape if size != 1
                    ])
                    # Reassign the squeezed tensor
                    model.set_initializer(node.input[1], new_shape)
                    # Track whether the shape actually changed
                    if len(new_shape) != len(shape):
                        # Is never reset back to False during iteration
                        graph_modified = True

            # Need to drop dimensions of size 1 from transpose permutation list
            if node.op_type == "Transpose":
                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(node.attribute, "perm")
                # If the permutation indices are given, we need to remove all
                # dimension of size 1 from these
                if perm is not None:
                    # Convert permutation indices to list of integers
                    perm = perm.ints
                    # Get the shape of the input tensor to seek for input
                    # dimensions of size 1
                    shape = model.get_tensor_shape(
                        node.input[0], fix_missing_init_shape=True
                    )
                    # Keep track of new axis enumeration, skipping dimensions of
                    # size 1
                    mapping, new_axis = {}, 0
                    # Enumerate the sizes per axis
                    for axis, size in enumerate(shape):
                        # Insert mapping from old to new axis
                        mapping[axis] = new_axis
                        # Only advance the new axis index for dimensions not to
                        # be squeezed
                        new_axis += (size > 1)
                    # Filter and remap the axis enumeration of the permutation
                    new_perm = [
                        mapping[axis] for axis in perm if shape[axis] > 1
                    ]
                    # Track whether the permutations actually changed
                    if len(new_perm) != len(perm) or new_perm != perm:
                        # Is never reset back to False during iteration
                        graph_modified = True
                    # Remove the permutation attribute before setting the new
                    # permutation
                    remove_by_name(node.attribute, "perm")
                    # Insert new permutation attribute
                    node.attribute.append(oh.make_attribute("perm", new_perm))
            # Need to the squeezed output mode of multi-head merging
            if node.op_type == "MergeMultiHeads":
                # Remove the squeezed attribute
                remove_by_name(node.attribute, "squeezed")
                # Set squeezed mode attribute
                node.attribute.append(oh.make_attribute("squeezed", True))
        # Iterate all tensors in the graph keeping track of the index
        for index, name in enumerate(model.get_all_tensor_names()):
            # Query the shape of the tensor adding annotations for initializers
            # if missing
            shape = model.get_tensor_shape(name, fix_missing_init_shape=True)
            # Skip squeezing 0d or 1d tensors
            if len(shape) <= 1:
                continue
            # Squeeze the shape by removing all dimensions with size 1
            new_shape = [size for size in shape if size != 1]
            # Try to get the initializer of the tensor
            initializer = model.get_initializer(name)
            # If an initializer is present replace by the squeezed tensor
            if initializer is not None:
                # Reassign the squeezed tensor
                model.set_initializer(name, initializer.squeeze())
            # Set new shape annotation
            model.set_tensor_shape(name, new_shape)
            # Track whether the shape actually changed
            if len(new_shape) != len(shape):
                # Is never reset back to False during iteration
                graph_modified = True
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Removes identity reshape operations, i.e., Reshape where input shape is the
# same as the target shape
class RemoveIdentityReshape(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Reuse node removal and rewiring from qonnx
        from qonnx.transformation.remove import remove_node_and_rewire
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Reshape operation types
            if node.op_type == "Reshape":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Second input to the reshape operation is the target shape
                shape = model.get_initializer(node.input[1])
                # If the initializer is present, this is a constant shape
                # reshape which can be removed if it does not reshape
                if shape is not None:
                    # Get the shape of the input to the reshape
                    inp = model.get_tensor_shape(node.input[0])
                    # If input and target shape are the same, this is an
                    # identity operation
                    if len(shape) == len(inp) and (shape == inp).all():  # noqa
                        # Remove and rewire this node
                        remove_node_and_rewire(model, node)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows from outer scope
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Removes identity transpose operations, i.e., Transpose where input order is
# the same as the target permutation
class RemoveIdentityTranspose(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Reuse node removal and rewiring from qonnx
        from qonnx.transformation.remove import remove_node_and_rewire
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Transpose operation types
            if node.op_type == "Transpose":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(node.attribute, "perm")
                # If the permutation indices are given, we need to remove all
                # dimension of size 1 from these
                if perm is not None:
                    # Convert permutation indices to list of integers
                    perm = perm.ints
                    # Get the shape of the input tensor
                    shape = model.get_tensor_shape(
                        node.input[0], fix_missing_init_shape=True
                    )
                    # If the permutation indices cover the input shape in order,
                    # this transpose does nothing
                    if perm == [i for i in range(len(shape))]:
                        # Remove and rewire this node
                        remove_node_and_rewire(model, node)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        # model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Script entrypoint
if __name__ == '__main__':
    # Load the model graph
    model = ModelWrapper("attention.onnx")

    # Add shape and datatype annotations throughout all the graph
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())

    # Cleanup the graph by removing redundant, unnecessary and constant nodes
    # and tensors and give unique names to everything remaining
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(FoldConstants())

    # # Removes dimension of size 1, i.e., the batch dimension
    # model = model.transform(Squeeze())
    # Remove unnecessary shape and layout transformations
    model = model.transform(RemoveIdentityReshape())
    model = model.transform(RemoveIdentityTranspose())
    # Insert tensor layout annotations for Quant tot MultiThreshold transform
    # to determine the correct output channel dimension
    model = model.transform(InferDataLayouts())

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
