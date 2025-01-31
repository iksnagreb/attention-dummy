# Python warning messages
import warnings
# Copies of python objects
from copy import deepcopy
# Numpy for handling tensors form the ONNX graphs
import numpy as np

# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Base class for all QONNX graph transformations and some basic cleanup
# transformations
from qonnx.transformation.general import (
    Transformation,
    ConvertDivToMul,
    ConvertSubToAdd,
)

# QONNX graph transformations for annotating the graph with datatype and shape
# information
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
# Converts BatchNorm operation to affine transformation
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine

# Groups node inputs by dynamic vs. initializer category
from finn.transformation.streamline.absorb import group_inputs_by_category

# FINN streamlining transformations converting and rounding values
from finn.transformation.streamline import (
    ConvertSignToThres,
    RoundAndClipThresholds
)
# FINN streamlining transformations reordering the graph
from finn.transformation.streamline.reorder import (
    MoveMulPastFork,
    MoveTransposePastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarLinearPastInvariants,
    MoveTransposePastEltwise,
    MoveMulPastMaxPool,
    MoveAddPastMul,
    MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastMatMul,
    MoveScalarMulPastConv,
    MoveTransposePastJoinMul,
    MoveTransposePastJoinAdd,
    MoveMulPastJoinAdd,
    MoveAddPastJoinAdd,
    MoveScalarLinearPastSplit,
    MoveAffinePastJoinConcat,
    MoveMulPastJoinConcat,
    MoveAddPastJoinConcat,
    MoveTransposePastSplit,
    MoveTransposePastJoinConcat,
    MoveSqueezePastMultiThreshold,
    is_scalar
)
# FINN streamlining transformations absorbing tensors/nodes into others
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    AbsorbMulIntoMultiThreshold,
    Absorb1BitMulIntoMatMul,
    Absorb1BitMulIntoConv,
    AbsorbTransposeIntoMultiThreshold
)
# FINN streamlining transformations fusing/collapsing operations of the same
# kind
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedMul,
    CollapseRepeatedTranspose,
    CollapseRepeatedAdd
)
# FINN streamlining transformations removing nodes without real effect from the
# graph
from finn.transformation.streamline.remove import (
    RemoveIdentityTranspose,
    RemoveIdentityReshape
)

# Custom transformation for exhaustively composing transformations
from .composed_transformation import ComposedTransformation


# Moves elementwise additions past MatMul operations: Applicable if each
# operation has one initializer input
class MoveAddPastMatMul(Transformation):
    # Applies the transform to a whole model graph  # noqa: Duplicate
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Add operations
            if node.op_type == "Add":
                # If the add is a join operation, we do not have a constant
                # added to the input
                if model.is_join_node(node):
                    # Skip transforming this
                    continue
                # If the Add is a fork operation we should first distribute the
                # Add into the branches
                if model.is_fork_node(node):
                    # Issue a warning to make the use aware of this potential
                    # transformation if the fork is moved first
                    warnings.warn(
                        f"{self.__class__.__name__}:"
                        f" Skipping near match: {node.name} is a fork-node,"
                        f" try MoveLinearPastFork first"
                    )
                    # Skip transforming this node as moving this would lead
                    # to messed up or detached graph
                    continue
                # Decompose the inputs into the dynamic and the constant
                # initializer input
                (x_name,), (c_name,) = group_inputs_by_category(node, model)
                # Now check the successor node which must be a MatMul
                consumer = model.find_direct_successors(node)
                # If there is no consumer, this Add seems to be last node of the
                # graph
                if not consumer:
                    # Skip transforming this
                    continue
                # There must be exactly one consumer now
                consumer = consumer[0]
                # This transformation only applies to Add in front of MatMul
                if not consumer.op_type == "MatMul":
                    # Skip this if not MatMul
                    continue
                # MatMul may not be a join operation to apply this
                # transformation
                if model.is_join_node(consumer):
                    # Skip transforming without warning (there is nothing we can
                    # do about this)
                    continue
                # Decompose the inputs to the MatMul to get the weight tensor
                # name (the other input is the output of the Add)
                _, (w_name,) = group_inputs_by_category(consumer, model)
                # Read the weights and the constant addition tensor
                w = model.get_initializer(w_name)
                c = model.get_initializer(c_name)
                # Determine whether the weights are the left or right input to
                # the MatMul
                left = w_name == consumer.input[0]
                # Apply the weights to the constant tensor
                c = np.matmul(w, c) if left else np.matmul(c, w)
                # Insert the transformed tensor back into the mode as an
                # initializer
                model.set_initializer(c_name, c)
                # The connecting tensors of this pattern
                inp = x_name
                mid = node.output[0]
                out = consumer.output[0]
                # Rewire the graph pattern connecting the input to the MatMul
                # and the MatMul output to the Add node
                consumer.input[1 if left else 0] = inp
                # The Add now produces the original MatMul output
                node.output[0] = out
                # The middel tensor connects to the Add input
                node.input[0 if node.input[0] == x_name else 1] = mid
                # The MatMul feeds the middle tensors
                consumer.output[0] = mid
                # Delete the shape annotation of the connecting tensors
                # to be re-done later
                model.set_tensor_shape(mid, None)
                model.set_tensor_shape(out, None)
                # Delete the type annotations of the connecting tensors
                # to be re-done later
                # model.set_tensor_datatype(mid, None)
                # model.set_tensor_datatype(out, None)
                # Track whether the graph has been modified, never
                # resets to False
                graph_modified = True
                # Break the loop after deleting shape annotations to
                # immediately re-do these before changing the next
                # operator
                break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves constant elementwise multiplication past another joining multiplication
class MoveConstMulPastJoinMul(Transformation):
    # Applies the transform to a whole model graph  # noqa: Duplicate
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to Multiplications
                if successor.op_type in {"Mul"}:
                    # Applies only if the second multiplication is a join-node
                    if model.is_join_node(successor):
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Need to match the correct input of the joining second
                        # multiplication
                        for i, name in enumerate(successor.input):
                            # If the successors input currently matches the
                            # intermediate tensors, this input needs to be
                            # rewired
                            if name == mid:
                                # Rewire the graph to feed original into the
                                # second Mul node first
                                successor.input[i] = inp
                                # Note: Do not break here as it is perfectly
                                # legal to connect the same tensor multiple
                                # times to different inputs
                        # Repurpose the middle tensor for the output of the
                        # second Mul
                        successor.output[0] = mid
                        # The first Mul operator now gets the middle tensor as
                        # its input
                        node.input[0] = mid
                        # The first Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves elementwise multiplication past elementwise addition if one input to
# each of the operators is a known constant
# Note: Reverse of MoveAddPastMul
class MoveMulPastAdd(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type == "Mul":
                # Currently does not handle fork- or join-nodes
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If Squeeze is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Applies to additions
                if successor.op_type in {"Add"}:
                    # The addition may not join as we need to know the second
                    # input
                    if not model.is_join_node(successor):
                        # Get the constant initializer tensors for both
                        # operations: y = s * x + b
                        _, s_name = group_inputs_by_category(node, model)
                        _, b_name = group_inputs_by_category(successor, model)
                        # Skip if either node has no constant initializer
                        if not s_name or not b_name:
                            # Skip without warning ok?
                            continue
                        # There must be exactly one constant per operations
                        assert len(s_name) == 1, \
                            f"To many constant inputs for {node}"
                        assert len(b_name) == 1, \
                            f"To many constant inputs for {successor}"
                        # Now read the initializer tensors
                        s = model.get_initializer(*s_name)
                        b = model.get_initializer(*b_name)
                        # Update the addition initializer according to the
                        # distributive law
                        model.set_initializer(*b_name, b / s)
                        # Get names of all tensors involved in connecting the
                        # nodes
                        inp = node.input[0]  # noqa: Duplicate
                        mid = node.output[0]
                        out = successor.output[0]
                        # Rewire the graph to feed original input into the
                        # Add node first
                        successor.input[0] = inp
                        # Repurpose the middle tensor for the output of the Add
                        successor.output[0] = mid
                        # The Mul operator now gets the middle tensor as its
                        # input
                        node.input[0] = mid
                        # Mul now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
                        # Break the loop after deleting shape annotations to
                        # immediately re-do these before changing the next
                        # operator
                        break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves scalar linear elementwise operations past fork nodes, applies to Add,
# Mul, Sub, Div, etc.
class MoveScalarLinearPastFork(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul-like and Add-like operation types
            if node.op_type in {"Add", "Sub", "Mul", "Div"}:
                # Only handles non-joining forks for now
                if not model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Only handles one forking output for now
                if len(node.output) > 1:
                    # Softly skip this node
                    continue
                # Test whether the node initializer is a scalar...
                if not is_scalar(model.get_initializer(node.input[1])):
                    # Softly skip this node
                    continue
                # We need to insert a replica of this operation in front of each
                # consumer node
                for consumer in model.find_direct_successors(node):
                    # Create an exact replica of this operator
                    copy = deepcopy(node)
                    # Insert a new unique tensor connecting the output of the
                    # copy to the consumer
                    copy.output[0] = model.make_new_valueinfo_name()
                    # The original node might be connecting to multiple inputs
                    # of the consumer...
                    for idx, inp in enumerate(consumer.input):
                        # Find each instance of connection from original node
                        if inp == node.output[0]:
                            # Rewire to connect to the replica
                            consumer.input[idx] = copy.output[0]
                    # Insert the new replica node into the graph
                    graph.node.insert(index + 1, copy)
                # Remove the original node from the graph
                graph.node.remove(node)
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Moves scale factor, i.e., scalar Mul and Div, past Im2Col (and Col2Im): These
# cannot be handled by MoveScalarLinearPastInvariants as potential padding makes
# Add-Im2Col not commute to Im2Col-Add
class MoveScalesPastIm2Col(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Mul operation types
            if node.op_type in {"Mul", "Div"}:
                # Cannot handle fork- or join-multiplications
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Softly skip this node
                    continue
                # Only handles one forking output for now
                if len(node.output) > 1:
                    # Softly skip this node
                    continue
                # The first input must be dynamically received from upstream
                if model.get_initializer(node.input[0]) is not None:
                    # Softly skip this node
                    continue
                # Test whether the node initializer is a scalar...
                if not is_scalar(model.get_initializer(node.input[1])):
                    # Softly skip this node
                    continue
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)
                # If this is the final operation in the graph, there might be no
                # successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Now there is exactly one successor which needs to be extracted
                # from the list
                successor = successor[0]
                # Handle both, Im2Col and the inverse Col2Im, as well as padding
                if successor.op_type in {"Im2Col", "Col2Im", "Pad"}:
                    # Get names of all tensors involved in connecting the
                    # nodes
                    inp = node.input[0]  # noqa: Duplicate
                    mid = node.output[0]
                    out = successor.output[0]
                    # Rewire the graph to feed original input into the
                    # Add node first
                    successor.input[0] = inp
                    # Repurpose the middle tensor for the output of the Add
                    successor.output[0] = mid
                    # The Mul operator now gets the middle tensor as its
                    # input
                    node.input[0] = mid
                    # Mul now produces the original output tensor
                    node.output[0] = out
                    # Delete the shape annotation of the connecting tensors
                    # to be re-done later
                    model.set_tensor_shape(mid, None)
                    model.set_tensor_shape(out, None)
                    # Track whether the graph has been modified, never
                    # resets to False
                    graph_modified = True
                    # Break the loop after deleting shape annotations to
                    # immediately re-do these before changing the next
                    # operator
                    break
        # Redo datatype and shape annotations
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the transformation
        # needs to be applied again
        return model, graph_modified


# Define a set of custom streamlining transformations: These are applied once
# during the actual streamlining step and once after converting attention to
# hardware (the associated cleanup afterward might enable some Streamlining
# transformations once again)
def Streamline():  # noqa: Uppercase
    # Return a set of exhaustively applies transformations
    return ComposedTransformation([
        # On skip-connections: prefer pushing scalar multiplication forward
        # before MoveAddPastMul
        MoveMulPastFork(),
        # The "standard" set of FINN streamlining transformations or at least
        # inspired by them but applied exhaustively until none of them changes
        # the graph anymore.
        # Note: Covers most parts of non-branching linear topologies
        ComposedTransformation([
            ConvertSubToAdd(),
            ConvertDivToMul(),
            BatchNormToAffine(),
            ConvertSignToThres(),
            MoveMulPastMaxPool(),
            AbsorbSignBiasIntoMultiThreshold(),
            MoveScalarLinearPastInvariants(),
            MoveAddPastMul(),
            MoveScalarAddPastMatMul(),
            MoveAddPastConv(),
            MoveScalarMulPastMatMul(),
            MoveScalarMulPastConv(),
            MoveAddPastMul(),
            CollapseRepeatedAdd(),
            CollapseRepeatedMul(),
            MoveMulPastMaxPool(),
            AbsorbAddIntoMultiThreshold(),
            FactorOutMulSignMagnitude(),
            AbsorbMulIntoMultiThreshold(),
            Absorb1BitMulIntoMatMul(),
            Absorb1BitMulIntoConv(),
        ]),
        # Streamlining scales and biases forward through residual topologies
        # Note: This mostly covers forking and joining operations
        ComposedTransformation([
            # Note: This is probably the most common way of joining skip
            # connections, i.e., this corresponds to the original residual
            # addition, i.e., y = f(x) + x
            MoveLinearPastEltwiseAdd(),
            MoveScalarLinearPastFork(),
            MoveScalarLinearPastInvariants(),
            MoveMulPastFork(),
            MoveMulPastJoinAdd(),
            MoveAddPastJoinAdd(),
            # Note: This brings constant Muls (i.e., quantizer scales to be
            # removed) forward through joining Muls (i.e., those ending up
            # as actual hardware operators).
            MoveConstMulPastJoinMul()
        ]),
        # Streamlining scales and biases forward through shape/layout changing
        # operations, i.e., mostly transposes
        ComposedTransformation([
            # Convolution inputs and padding
            MoveScalesPastIm2Col(),
            # Streamlining for Split and Concat operations
            MoveScalarLinearPastSplit(),
            MoveAffinePastJoinConcat(),
            MoveMulPastJoinConcat(),
            MoveAddPastJoinConcat(),
            # Move transposes around to some place where they could be removed
            # later, i.e., where they collapse into identities
            MoveTransposePastFork(),
            MoveTransposePastSplit(),
            MoveTransposePastJoinConcat(),
            MoveTransposePastEltwise(),
            MoveTransposePastJoinMul(),
            MoveTransposePastJoinAdd(),
            CollapseRepeatedTranspose(),
            # Remove identity shape/layout transformations
            RemoveIdentityTranspose(),
            RemoveIdentityReshape(),
            # Squeeze operators can be moved past the thresholding
            MoveSqueezePastMultiThreshold(),
            # A certain type of 4d-layout transpose can be absorbed (actually
            # moved past) MultiThreshold operations
            AbsorbTransposeIntoMultiThreshold(),
        ]),
        # Only round and clip after all streamlining transformations have
        # been applied exhaustively.
        # Note: Might still enable another round of streamlining.
        RoundAndClipThresholds(),
    ])
