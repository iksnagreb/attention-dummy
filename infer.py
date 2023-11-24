# Output warning messages
import warnings
# Standard math functions
import math
# Make copies and deep copies of python objects
import copy
# Need numpy for modifying the onnx graph tensors, which are numpy style arrays
import numpy as np
# Protobuf onnx graph node type
from onnx import NodeProto  # noqa
# Helper for creating ONNX nodes
from onnx import helper as oh  # noqa
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX custom operations
from qonnx.custom_op.base import CustomOp
# QONNX datatypes
from qonnx.core.datatype import BaseDataType
# QONNX graph transformation base class
from qonnx.transformation.base import Transformation
# Transformation running onnx shape inference
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import (
    GiveUniqueNodeNames, GiveReadableTensorNames
)
# Gets items from protobuf by name
from qonnx.util.basic import get_by_name


# Tests whether a node is a join-node MatMul operation, i.e., a MatMul with two
# runtime inputs but no weights initializers
def is_join_matmul(node: NodeProto, model: ModelWrapper):  # noqa
    # Only handle existing MatMul type nodes
    if node is not None and node.op_type in {"MatMul"}:
        # No input must have an initializer
        return all(model.get_initializer(i) is None for i in node.input)
    # Did not match the operator type
    return False


# Tests whether a node is a MatMul operator
def is_matmul(node: NodeProto):
    # Node must exist and be of type MatMul
    return node is not None and node.op_type in {"MatMul"}


# Tests whether a node is a Softmax operator
def is_softmax(node: NodeProto):
    # Node must exist and be of type Softmax
    return node is not None and node.op_type in {"Softmax"}


# Tests whether a node is an element-wise Mul
def is_mul(node: NodeProto):
    # Node must exist and be of type Mul
    return node is not None and node.op_type in {"Mul"}


def is_end(node: NodeProto, model: ModelWrapper):  # noqa
    return node is not None and not model.find_direct_predecessors(node)


# Follow all input branches of a node until reaching a matmul
def all_upstream_to_matmul(node: NodeProto, model: ModelWrapper):  # noqa
    # Check whether the node is either a matmul node or the end of the graph
    def is_matmul_or_end(n: NodeProto):
        return is_matmul(n) or is_end(n, model)

    # Enumerate all inputs and collect everything upstream until finding the
    # next matmul operation
    return (model.find_upstream(i, is_matmul_or_end, True) for i in node.input)


# Projects a list of ONNX graph nodes to the string representation of the
# operator types
def op_types(nodes: list[NodeProto]) -> list[str]:
    return [node.op_type if node is not None else "None" for node in nodes]


# Convert the operator pattern corresponding to scaled dot-product attention to
# the HLS custom operator node
class InferScaledDotProductAttention(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # This transformation is triggered by finding a join-node MatMul
            if is_join_matmul(node, model):
                # If there are more than two branches feeding the MatMul, this
                # is probably not attention, softly skip the node
                if len(node.input) != 2:
                    continue
                # Follow both branches upstream looking for the next MatMul
                lhs, rhs = all_upstream_to_matmul(node, model)
                # Exactly one of the branches is supposed to contain a Softmax
                # operation
                if ("Softmax" in op_types(lhs)) == ("Softmax" in op_types(rhs)):
                    # TODO: Near match. But what is this? just skip?
                    continue
                # By convention and following the equation, the left hand side
                # of attention is the attention matrix, i.e., the one containing
                # Softmax and terminating in a join-node MatMul
                if "Softmax" not in op_types(lhs):
                    # Softmax must currently be on the right hand side, swap the
                    # order
                    lhs, rhs = rhs, lhs
                # The left hand side, i.e, attention matrix must terminate in a
                # join-node MatMul involving the query and key input
                if not is_join_matmul(lhs[-1], model):
                    # TODO: Near match. But what is this? just skip?
                    continue
                # Get shapes of input tensors, expect the second inputs, i.e.,
                # the keys to be transposed
                qh, ql, qe = model.get_tensor_shape(lhs[-1].input[0])
                kh, ke, kl = model.get_tensor_shape(lhs[-1].input[1])
                # The input shapes of the two matmul inputs must be compatible,
                # i.e., they must have matching embedding dimension
                if (qh, True, qe) != (kh, True, ke):
                    # Issue a warning of near match of the supported attention
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Mismatch in head or embedding dim at {lhs[-1].name}: "
                        f" {(qh, ql, qe)} vs. {(kh, kl, ke)}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue
                # There must be a Transpose feeding the key input
                transpose = model.find_producer(lhs[-1].input[1])
                # The transform applies only to transpose with exactly one input
                if transpose is None or len(transpose.input) != 1:
                    # Issue a warning of near match of the supported attention
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Missing Transpose near {lhs[-1].name}: "
                        f" {op_types([transpose])[0]}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Skip this node if the transpose output forks into multiple
                # branches
                if model.is_fork_node(transpose):
                    # Issue a warning of near match of the supported attention
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Fork Transpose near {node.name}: {transpose.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # The input shape of the transpose must match the transpose
                # of the key matrix
                # @formatter:off
                assert model.get_tensor_shape(transpose.input[0]) == [
                    kh, kl, ke
                ]
                # @formatter:on
                # Collect the input tensors to the attention operation, i.e.,
                # the query, key and value tensors
                q, k, v = lhs[-1].input[0], transpose.input[0], rhs[0].output[0]
                # Validate that the values are actually consumed by the final
                # matmul. For queries and keys this should all be given, as we
                # just walked upwards the graph.
                assert node in model.find_consumers(v)

                # Get the (optional) Softmax activation function
                act_a_softmax = lhs[0] if is_softmax(lhs[1]) else None
                # Get the (optional) query-key matmul activation function
                act_qk_matmul = lhs[-2] if is_matmul(lhs[-1]) else None

                # There might be no activation function between qk matmul and
                # softmax normalization
                if is_mul(act_qk_matmul) or is_softmax(act_qk_matmul):
                    # Remove the detected activation function node from the
                    # pattern candidates
                    act_qk_matmul = None

                # Check whether the node is a supported type of activation
                def is_supported_activation(n: NodeProto):  # noqa: Shadows name
                    # Currently, only none-type and MultiThreshold activations
                    # are supported
                    return n is None or n.op_type in {"MultiThreshold"}

                # Get the (optional) output matmul activation function
                act_av_matmul = model.find_direct_successors(node)
                # If the final matmul is a fork node, this needs to be handled
                # separately
                if act_av_matmul is not None and len(act_av_matmul) > 1:
                    # Assume no activation in this case
                    act_av_matmul = [None]
                # Unwrap the output activation from the list
                act_av_matmul, = act_av_matmul
                # The final activation can be omitted if it is not supported as
                # it might just be part of the next operator pattern
                if not is_supported_activation(act_av_matmul):
                    # Remove by setting to None (will be ignored by the next
                    # steps)
                    act_av_matmul = None
                # List all activations for validation and further processing
                #   Note: Order matters!
                acts = [act_qk_matmul, act_av_matmul, act_a_softmax]
                # Skip this node if any activation is not supported
                if not all(is_supported_activation(act) for act in acts):
                    # Issue a warning of near match of the supported attention
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Unsupported activation near {node.name}: "
                        f" One of {', '.join(op_types(acts))}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Check whether there is a de-quantizer scale factor preceding
                # the Softmax operator
                dequant_softmax = lhs[2] if is_softmax(lhs[1]) else None

                # Currently, only elementwise Mul is supported as de-quantizer
                if not is_mul(dequant_softmax):
                    # Issue a warning of near match of the supported attention
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Unsupported de-quantizer near {lhs[1].name}: "
                        f" {dequant_softmax.op_type}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # The last node of the attention operator is either the detected
                # matmul or the following, optional activation function
                last = act_av_matmul if act_av_matmul is not None else node

                # Tensor names of the threshold inputs
                # Note: order matters
                thresholds = [
                    # TODO: Fix condition once more activation types are
                    #  supported, currently there are only none and thresholds
                    act.input[1] for act in acts if act is not None
                ]

                # Convert activation function types to string representation
                def act_op_type_str(act):
                    # Only MultiThreshold is supported currently
                    if act is not None and act.op_type == "MultiThreshold":
                        # The attention custom op uses "thresholds" to identify
                        return "thresholds"
                    # All other types are not supported
                    return "none"

                # The value tensor shape must be compatible with the attention
                # matrix
                assert model.get_tensor_shape(v)[:2] == [qh, kl]

                # Fixed node attributes and extracted input/output/initializer
                # tensor names
                kwargs = {
                    # Refer to this operator type by its name
                    "op_type": "ScaledDotProductAttention",
                    # Execution will try to look up the implementation in the
                    # package
                    # referred to by the domain
                    "domain": "finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    "backend": "fpgadataflow",
                    # Named inputs and activation thresholds extracted from the
                    # graph pattern
                    # TODO: Currently no masking support
                    "inputs": [q, k, v, *thresholds, dequant_softmax.input[1]],
                    # Named model output extracted from the graph pattern
                    "outputs": last.output,
                    # TODO: Currently no masking support
                    "mask_mode": "none",
                    # Give node name derived from the operator type and the name
                    # of the triggering node to be removed
                    "name": f"ScaledDotProductAttention_{node.name}"
                }

                # Extract the node attributes of the attention operator from
                # all constituent nodes
                node_attrs = {
                    # Number of attention heads
                    "Heads": qh,
                    # Embedding dimension of queries and keys
                    "QKDim": qe,
                    # Length of the query sequence
                    "QLen": ql,
                    # Embedding dimension of the values
                    "VDim": model.get_tensor_shape(v)[2],
                    # Length of the key and value sequence
                    "KVLen": kl,

                    # Folding along the embedding dimensions
                    # Note: Assume biggest folding possible fitting both
                    # embedding dimensions
                    "EmbFold": math.gcd(qe, model.get_tensor_shape(v)[2]),
                    # Folding along the sequence dimensions
                    # Note: Assume biggest folding possible fitting both
                    # sequence dimensions
                    "SeqFold": math.gcd(ql, kl),

                    # Datatype of query matrix elements
                    "QType": model.get_tensor_datatype(q),
                    # Datatype of key matrix elements
                    "KType": model.get_tensor_datatype(k),
                    # Datatype of value matrix elements
                    "VType": model.get_tensor_datatype(v),
                    # # Datatype of mask matrix elements
                    "MType": "UINT1",
                    # Datatype of attention weights elements
                    "AType": model.get_tensor_datatype(lhs[0].output[0]),
                    # Datatype of output elements
                    "OType": model.get_tensor_datatype(last.output[0]),

                    # Datatype of accumulator elements of the first matmul
                    "AccQKMatMul": model.get_tensor_datatype(lhs[-1].output[0]),
                    # Datatype of output elements of the first matmul
                    # Note: Can be extracted from the left hand side
                    # intermediate outputs
                    "OutQKMatMul": model.get_tensor_datatype(
                        # TODO: Clean this up...
                        act_qk_matmul.output[
                            0] if act_qk_matmul is not None else lhs[-1].output[
                            0]
                    ),
                    # Activation function type following the first matmul
                    "ActQKMatMul": act_op_type_str(act_qk_matmul),

                    # Datatype of accumulator elements of the second matmul
                    "AccAVMatMul": model.get_tensor_datatype(node.output[0]),
                    # Datatype of output elements of the second matmul
                    # Note: Always the same as the OType
                    "OutAVMatMul": model.get_tensor_datatype(last.output[0]),
                    # Activation function type following the second matmul
                    "ActAVMatMul": act_op_type_str(act_av_matmul),

                    # Activation function type following the softmax
                    # normalization of the attention weights
                    "ActASoftmax": act_op_type_str(act_a_softmax),

                    # Softmax may be preceded by a de-quantizer scalar
                    # multiplication
                    "DequantSoftmax": dequant_softmax.input[1]
                }

                # Converts QONNX datatypes to their name (as a string)
                def maybe_name(value):
                    # All QONNX datatypes are instances of the BaseDataType
                    if isinstance(value, BaseDataType):
                        # Convert to the name by referring to the datatypes name
                        # attribute
                        return value.name
                    # Everything else is just assumed to be in the right format
                    return value

                # Convert all node attributes DataTypes to string
                # representations of their names
                node_attrs = {
                    key: maybe_name(value) for key, value in node_attrs.items()
                }

                # Create a new custom node replacing the scaled dot-product
                # attention pattern
                attention = oh.make_node(**kwargs, **node_attrs)
                # Insert the new node into the graph
                graph.node.insert(index, attention)
                # Collect all nodes comprising the original pattern
                nodes = [node, transpose, *lhs, act_av_matmul]
                # Remove all nodes of the original pattern
                for n in nodes:
                    # Do not try to remove non-existing nodes
                    if n is not None:
                        graph.node.remove(n)
                # The graph has been modified
                graph_modified = True
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Tests whether a node is a Reshape operator
def is_reshape(node: NodeProto):
    return node is not None and node.op_type in {"Reshape"}


# Tests whether a node is a Transpose operator
def is_transpose(node: NodeProto):
    return node is not None and node.op_type in {"Transpose"}


# Tests whether a node is a Reshape-Transpose operator chain
def is_reshape_transpose(node: NodeProto, model: ModelWrapper):  # noqa
    # Reshape-transpose pattern detection is triggered by detecting a reshape
    # operation
    if is_reshape(node):
        # The reshape may not be a join or fork node
        if model.is_join_node(node) or model.is_fork_node(node):
            # Reject detection of the pattern
            return False
        # Get the single successor node
        transpose = model.find_direct_successors(node)[0]
        # The consumer must be Transpose finalizing the reshaping
        if not is_transpose(transpose):
            # Reject detection of the pattern
            return False
        # The transpose may not fork or join either
        if model.is_join_node(transpose) or model.is_fork_node(transpose):
            # Reject detection of the pattern
            return False
        # Accept detecting the pattern
        return True
    # Reject detection of the pattern
    return False


# Tests whether a node is a Transpose-Reshape operator chain
def is_transpose_reshape(node: NodeProto, model: ModelWrapper):  # noqa
    # Transpose-Reshape pattern detection is triggered by detecting a transpose
    # operation
    if is_transpose(node):
        # The transpose may not be a join or fork node
        if model.is_join_node(node) or model.is_fork_node(node):
            # Reject detection of the pattern
            return False
        # Get the single successor node
        reshape = model.find_direct_successors(node)[0]
        # The consumer must be a reshape finalizing the transpose-reshape
        if not is_reshape(reshape):
            # Reject detection of the pattern
            return False
        # The reshape may not fork or join either
        if model.is_join_node(reshape) or model.is_fork_node(reshape):
            # Reject detection of the pattern
            return False
        # Accept detecting the pattern
        return True
    # Reject detection of the pattern
    return False


# Infers reshaping of attention heads
class InferMultiHeads(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Head-slicing reshaping is triggered by detecting a reshape
            # operation followed by a transpose
            if is_reshape_transpose(node, model):
                # Get the single successor node
                transpose = model.find_direct_successors(node)[0]

                # Get the input and output tensor names to the pattern
                inp = node.input[0]
                mid = node.output[0]
                end = transpose.output[0]

                # Get the shape of the input tensor for inferring the number of
                # heads and correctly propagating shapes
                shape = model.get_tensor_shape(inp)
                # Determine the rank of the input tensor to support batched and
                # non-batched inputs
                rank = len(shape)

                # The input shape determines the sequence length
                seq, _, dim = shape if (rank == 3) else (shape[0], 1, shape[1])

                # The intermediate shape must be the same as specified as the
                # second input to the reshape operation
                assert (model.get_tensor_shape(mid)  # noqa
                        == model.get_initializer(node.input[1])).all()  # noqa
                # Expected layout after reshape is "head last"
                _, heads, _ = model.get_tensor_shape(mid)

                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(transpose.attribute, "perm")
                # Convert permutation indices to list of integers if it is
                # given
                perm = perm.ints if perm is not None else None

                # Transpose must either keep or flip the sequence and embedding
                # dimensions
                if perm not in [[1, 0, 2], [1, 2, 0]]:
                    # Issue a warning of near match of the supported head
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Unsupported permutation near {transpose.name}: {perm}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Check whether the transpose only permutes to head first or
                # additionally transposes sequence and embedding dimension as
                # well
                keep_transpose = (perm == [1, 2, 0])

                # Start assuming there is no middle node, as the transpose is
                # removed
                maybe_mid = end

                # Insert a new transpose node if the sequence and embedding
                # dimensions are flipped
                if keep_transpose:
                    # Construct a new intermediate tensor using the current one
                    # as template
                    maybe_mid = mid
                    # Construct a new Transpose with attributes inferred from
                    # the detected graph patter
                    new_transpose = oh.make_node(**{
                        "op_type": "Transpose",
                        # Named inputs extracted from the graph pattern
                        "inputs": [maybe_mid],
                        # Named outputs extracted from the graph pattern
                        "outputs": [end],
                        # Give node name derived from the operator type and the
                        # name of the triggering node to be removed
                        "name": f"MultiHeads_Transpose_{node.name}",
                        # Permute the last two dimensions
                        "perm": [0, 2, 1]
                    })
                    # Insert the new node into the graph
                    graph.node.insert(index + 1, new_transpose)
                    # Change the shape of the intermediate tensor to reflect
                    # partial reshaping
                    model.set_tensor_shape(
                        maybe_mid, (heads, seq, dim // heads)
                    )

                # Fixed node attributes and extracted input/output/initializer
                # tensor names
                kwargs = {
                    # Refer to this operator type by its name
                    "op_type": "SliceMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    "domain": "qonnx.custom_op.general",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    "backend": "fpgadataflow",
                    # Named inputs extracted from the graph pattern
                    "inputs": [inp],
                    # Named outputs extracted from the graph pattern
                    "outputs": [maybe_mid],
                    # Give node name derived from the operator type and the name
                    # of the triggering node to be removed
                    "name": f"SliceMultiHeads_{node.name}",
                    # Number of attention heads inferred
                    "heads": heads,
                    # Inferred multi-heads produce packed tensors
                    "packed": True
                }

                # Create a new custom node replacing the multi head reshape
                heads = oh.make_node(**kwargs)
                # Insert the new node into the graph
                graph.node.insert(index, heads)
                # Collect all nodes comprising the original pattern
                nodes = [node, transpose]
                # Remove all nodes of the original pattern
                for n in nodes:
                    # Do not try to remove non-existing nodes
                    if n is not None:
                        graph.node.remove(n)
                # The graph has been modified
                graph_modified = True

            # Head-merging reshaping is triggered by detecting a transpose
            # operation followed by a reshape
            if is_transpose_reshape(node, model):
                # Get the single successor node
                reshape = model.find_direct_successors(node)[0]

                # Get the input and output tensor names to the pattern
                inp = node.input[0]
                end = reshape.output[0]

                # The input shape determines the heads, sequence length and
                # embedding dimension
                heads, seq, dim = model.get_tensor_shape(inp)

                # Get the (optional) permutation indices of the transpose in
                # case it is a multi-axis transpose
                perm = get_by_name(node.attribute, "perm")
                # Convert permutation indices to list of integers if it is given
                perm = perm.ints if perm is not None else None

                # Transpose must flip the heads and sequence dimensions
                if perm not in [[1, 0, 2]]:
                    # Issue a warning of near match of the supported head
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Unsupported permutation near {node.name}: {perm}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Shape of the final output of the operator pattern
                out_shape = model.get_tensor_shape(end)

                # The output of the reshape must be the same as specified as the
                # second input to the reshape operation
                assert (out_shape  # noqa
                        == model.get_initializer(reshape.input[1])).all()

                # The final output shape must match the expectation of
                # reintegrating the heads back into the embeddings
                if out_shape not in [[seq, heads * dim], [seq, 1, heads * dim]]:
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Output shape mismatch near: {reshape.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Fixed node attributes and extracted input/output/initializer
                # tensor names
                kwargs = {
                    # Refer to this operator type by its name
                    "op_type": "MergeMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    "domain": "qonnx.custom_op.general",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    "backend": "fpgadataflow",
                    # Named inputs extracted from the graph pattern
                    "inputs": [inp],
                    # Named outputs extracted from the graph pattern
                    "outputs": [end],
                    # Give node name derived from the operator type and the name
                    # of the triggering node to be removed
                    "name": f"MergeMultiHeads_{node.name}",
                    # Number of attention heads inferred
                    "heads": heads,
                    # Remember, whether the output needs to be squeezed
                    "squeezed": out_shape == [seq, heads * dim],
                    # Inferred multi-heads produce packed tensors
                    "packed": True
                }

                # Create a new custom node replacing the multi head reshape
                heads = oh.make_node(**kwargs)
                # Insert the new node into the graph
                graph.node.insert(index, heads)
                # Collect all nodes comprising the original pattern
                nodes = [node, reshape]
                # Remove all nodes of the original pattern
                for n in nodes:
                    # Do not try to remove non-existing nodes
                    if n is not None:
                        graph.node.remove(n)
                # The graph has been modified
                graph_modified = True
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows from outer scope
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# QONNX custom op corresponding to SliceMultiHeads to support shape inference
# and node execution
class SliceMultiHeads(CustomOp):
    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        return {
            # Number of attention heads
            "heads": ("i", True, 1),
            # Specifies whether the output is packed as a single output tensor
            # or split as multiple output tensors
            "packed": ("i", True, 1)
        }

    # Makes an operation compatible with the output shape for shape inference
    #   Note: Propagates shape forward, i.e., never asks for the shape of the
    #   output, even if it seems easier.
    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the number of attention heads
        heads = self.get_nodeattr("heads")
        # Shape inference differs depending on packed or split outputs
        packed = self.get_nodeattr("packed")
        # Get the shape of the input tensor for inferring the number of
        # heads and correctly propagating shapes
        shape = model.get_tensor_shape(node.input[0])
        # Determine the rank of the input tensor to support batched and
        # non-batched inputs
        rank = len(shape)
        # The input shape determines the sequence length
        seq, _, dim = shape if (rank == 3) else (shape[0], 1, shape[1])
        # Packed outputs a represented by a reshape operation producing one
        # tensor
        if packed:
            # Create a new name for the temporary shape tensor
            shape = model.make_new_valueinfo_name()
            # Set the target shape of slices heads
            model.set_initializer(shape, np.asarray([heads, seq, dim // heads]))
            # Return a node simulating the shape effect of slicing into
            # multi-heads
            return oh.make_node(
                "Reshape", [node.input[0], shape], [node.output[0]]
            )
        # Prepare a dummy input to simulate reordering of batch/head dimension
        # to the front
        mock_input = model.make_new_valueinfo_name()
        # Set the target shape of slices heads
        model.set_tensor_shape(
            mock_input, [1, seq, dim] if rank == 3 else [seq, dim]
        )
        # If the outputs are not packed, the operation is represented as a split
        # operation producing number of heads outputs along the last axis
        return oh.make_node(
            "Split", [mock_input], node.output, num_outputs=heads, axis=-1
        )

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Propagate the type from the input to each output tensor
        for o in node.output:
            # Slicing simply propagates the type of the input to the output
            model.set_tensor_datatype(
                o, model.get_tensor_datatype(node.input[0])
            )

    # Executes multi-head slicing in python
    def execute_node(self, context, graph):
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the number of attention heads
        heads = self.get_nodeattr("heads")
        # Get the input out of the execution context
        #   Note: Shape must be either seq x 1 x dim or seq x dim
        inp = context[node.input[0]]
        # Execution differs depending on packed or split outputs, i.e., produce
        # one reshaped tensor vs. multiple split tensors
        packed = self.get_nodeattr("packed")
        # Packed execution boils down to a reshape of the single input to a
        # single output
        if packed:
            # Reshape to separate the heads out of the embedding dimensions,
            # finally transpose to heads first layout
            out = inp.reshape(inp.shape[0], heads, -1).transpose(1, 0, 2)
            # Write the output into the execution context
            context[node.output[0]] = out
        # Split is realized as the split operation of numpy
        else:
            # Produces multiple outputs as a list
            splits = np.split(inp, indices_or_sections=heads, axis=-1)
            # Correspondence between outputs and splits in order
            for o, out in zip(node.output, splits):
                # Write the output into the execution context
                context[o] = out

    # Verifies the node attributes, inputs and outputs
    def verify_node(self):
        # TODO: Implement
        return []


# QONNX custom op corresponding to MergeMultiHeads to support shape inference
# and node execution
class MergeMultiHeads(CustomOp):
    # Defines attributes which must be present on this node
    def get_nodeattr_types(self):
        return {
            # Number of attention heads
            "heads": ("i", True, 1),
            # Output needs to be squeezed
            "squeezed": ("i", True, 0),
            # Specifies whether the input is packed as a single input tensor
            # or split as multiple input tensors
            "packed": ("i", True, 1)
        }

    # Makes an operation compatible with the output shape for shape inference
    #   Note: Propagates shape forward, i.e., never asks for the shape of the
    #   output, even if it seems easier.
    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the number of attention heads
        heads = self.get_nodeattr("heads")
        # Squeeze single-element batch dimension from the output?
        squeezed = self.get_nodeattr("squeezed")
        # Shape inference differs depending on packed or split outputs
        packed = self.get_nodeattr("packed")
        # Packed inputs a represented by a reshape operation consuming one
        # tensor
        if packed:
            # Get the shape of the input tensor for inferring the number of
            # heads and correctly propagating shapes
            h, seq, dim = model.get_tensor_shape(node.input[0])
            # Attribute heads must match wht is annotated at the input
            assert h == heads, \
                f"Shape annotation and number of heads differ: {node.name}"
            # Distribute the heads into the embedding dimension
            dim = heads * dim
            # Create a new name for the temporary shape tensor
            shape = model.make_new_valueinfo_name()
            # Set the target shape of slices heads
            model.set_initializer(
                shape, np.asarray([seq, dim] if squeezed else [seq, 1, dim])
            )
            # Return a node simulating the shape effect of merging multi-heads
            return oh.make_node(
                "Reshape", [node.input[0], shape], [node.output[0]]
            )
        # If the inputs are not packed, the operation is represented as a concat
        # operation consuming number of heads inputs concatenating along the
        # last axis
        return oh.make_node("Concat", node.input, node.output, axis=-1)

    # Infers the datatype of the node output
    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Merging simply propagates the type of the input to the output
        model.set_tensor_datatype(
            node.output[0], model.get_tensor_datatype(node.input[0])
        )

    # Executes multi-head merging in python
    def execute_node(self, context, graph):
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the input out of the execution context
        #   Note: Shape must be heads x seq x dim
        inp = context[node.input[0]]
        # Get the number of attention heads
        heads = self.get_nodeattr("heads")
        # Execution differs depending on packed or split inputs, i.e., expects
        # one tensor vs. multiple split tensors
        packed = self.get_nodeattr("packed")
        # Packed execution boils down to a reshape of the single input to a
        # single output
        if packed:
            # Transpose back into sequence first layout then reintegrate the
            # heads via reshape
            out = inp.transpose(1, 0, 2).reshape(
                inp.shape[1], 1, heads * inp.shape[-1]
            )
        # Split is realized as the concat operation of numpy
        else:
            # Collect the list of inputs from the execution context and
            # concatenate along the last axis
            out = np.concatenate([context[i] for i in node.input], axis=-1)
            # Reshape to simulate the batch dimensions if it is not present
            out = out.reshape(out.shape[0], 1, out.shape[-1])
        # Optionally squeeze the output (remove batch dimension of size 1)
        if self.get_nodeattr("squeezed"):
            # Squeeze batch dimension via reshape
            out = out.reshape(out.shape[0], out.shape[-1])
        # Write the output into the execution context. Force output shape
        # which might be squeezed
        context[node.output[0]] = out

    # Verifies the node attributes, inputs and outputs
    def verify_node(self):
        # TODO: Implement
        return []


# Transplant the new custom ops into the QONNX domain
import qonnx.custom_op.general  # noqa

qonnx.custom_op.general.custom_op["SliceMultiHeads"] = SliceMultiHeads
qonnx.custom_op.general.custom_op["MergeMultiHeads"] = MergeMultiHeads


# Move SliceMultiHeads operation past MultiThreshold operation
class MoveSliceMultiHeadsPastMultiThreshold(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Transformation applies to SliceMultiHeads operation (not Merge)
            if node.op_type == "SliceMultiHeads":
                # Slicing should not fork or join
                if model.is_fork_node(node) or model.is_join_node(node):
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Slicing may not join or fork: {node.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue
                # Now we know there is only one consumer operation following the
                # slice node
                thresholds_node = model.find_direct_successors(node)[0]
                # Successor must actually be a MultiThresholds for this
                # transform to apply
                if not thresholds_node.op_type == "MultiThreshold":
                    # Skip transforming this instance, probably no need to warn
                    continue

                # Thresholds should not fork or join either
                if (model.is_fork_node(thresholds_node)
                        or model.is_join_node(thresholds_node)):
                    # Issue a warning to make the user aware of this mismatch
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"MultiThreshold may not join or fork:"
                        f" {thresholds_node.name}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue

                # Get the thresholds tensor, which must be an initializer at
                # the second input
                thresholds = model.get_initializer(thresholds_node.input[1])
                # This is indeed an error, no way to recover from this, so
                # assertion is fine
                assert thresholds is not None, \
                    f"Missing threshold tensor for {thresholds_node.name}"

                # The slice node should have an attribute specifying the number
                # of heads
                heads = get_by_name(node.attribute, "heads")
                # Heads must be present, otherwise this is an errr
                assert heads is not None, \
                    f"Missing number of heads for {node.name}"
                # Convert heads attribute proto to integer
                heads = heads.i

                # Repeat the thresholds for each head along the channel
                # dimension
                thresholds = np.concatenate(heads * [thresholds])
                # Update the thresholds tensor to simply repurpose the existing
                # node
                model.set_initializer(thresholds_node.input[1], thresholds)

                # Get names of all tensors involved in connecting the nodes
                inp = node.input[0]
                mid = node.output[0]
                out = thresholds_node.output[0]

                # The middle tensor is now produced by the multi-threshold,
                # which does not change the shape. Propagate the shape of the
                # input tensor
                model.set_tensor_shape(mid, model.get_tensor_shape(inp))
                # As the middle tensor is now produced by the multi-threshold,
                # the datatype needs to be taken from the output tensor
                model.set_tensor_datatype(mid, model.get_tensor_datatype(out))

                # Rewire the nodes locally switching order. Reuses all the
                # exising tensors.
                thresholds_node.input[0] = inp
                thresholds_node.output[0] = mid
                node.input[0] = mid
                node.output[0] = out

                # Graph has been modified, required additional transformations
                # to be run
                graph_modified = True
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows from outer scope
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Detects multi-head attention pattern, i.e., scaled dot-product attention
# between head slicing and merging
def is_multi_head_attention(node: NodeProto, model: ModelWrapper):  # noqa
    # The anchor node must be scaled dot product attention
    if node.op_type == "ScaledDotProductAttention":
        # Get the nodes feeding the attention operation
        predecessors = model.find_direct_predecessors(node)
        # There must be exactly three predecessors of type head-slicing
        # Note: the may be nothing in between slicing and the attention itself
        if op_types(predecessors) == 3 * ["SliceMultiHeads"]:
            # Get the node fed by the attention operation
            successors = model.find_direct_successors(node)
            # There must be exactly onde successor of type head-merging
            # Note: the may be nothing in between attention and the merging
            if op_types(successors) == 1 * ["MergeMultiHeads"]:
                # Get the shape of the input tensor for inferring the number of
                # heads and correctly propagating shapes
                shape = model.get_tensor_shape(node.input[0])
                # Determine the rank of the input tensor to support batched and
                # non-batched inputs
                rank = len(shape)
                # The input shape determines the sequence length
                heads, _, _ = shape if (rank == 3) else (1, shape[0], shape[1])
                # Pattern detected, if there are actually multiple heads
                return heads > 1
    # Pattern not detected
    return False


# Unrolls multiple attention heads in the onnx graph
class UnrollMultiHeadAttention(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Apply transformation to node which match the multi-head attention
            # pattern
            if is_multi_head_attention(node, model):
                # Get the slicing nodes fed by the attention operation
                slice0, slice1, slice2 = model.find_direct_predecessors(node)
                # Get the single merging node
                merge0, = model.find_direct_successors(node)
                # Get the number of heads produced by an arbitrary slicer
                heads = get_by_name(slice0.attribute, "heads").i
                # Validate the number of heads matches between all slice and
                # merge nodes
                for n in [slice0, slice1, slice2, merge0]:
                    # All heads must match, otherwise this is a failure from
                    # which we cannot recover
                    assert get_by_name(n.attribute, "heads").i == heads, \
                        f"Differing number of heads at {node.name} and {n.name}"
                    # Remove the original node from the graph
                    graph.node.remove(n)

                # TODO: Clean up the following code

                # Create replicas of the slicing nodes with expanded output list
                slice0 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="SliceMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="qonnx.custom_op.general",
                    # Connect to the same input as the original
                    inputs=slice0.input,
                    # Generate new output tensor names for each head
                    outputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Unrolled heads do not produce packed tensors
                    packed=False
                )
                slice1 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="SliceMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="qonnx.custom_op.general",
                    # Connect to the same input as the original
                    inputs=slice1.input,
                    # Generate new output tensor names for each head
                    outputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Unrolled heads do not produce packed tensors
                    packed=False
                )
                slice2 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="SliceMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="qonnx.custom_op.general",
                    # Connect to the same input as the original
                    inputs=slice2.input,
                    # Generate new output tensor names for each head
                    outputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Unrolled heads do not produce packed tensors
                    packed=False
                )
                # Create replica of the merging node with expanded input list
                merge0 = oh.make_node(
                    # Refer to this operator type by its name
                    op_type="MergeMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    domain="qonnx.custom_op.general",
                    # Generate new input tensor names for each head
                    inputs=[
                        model.make_new_valueinfo_name() for _ in range(heads)
                    ],
                    # Connect to the same input as the original
                    outputs=merge0.output,
                    # Attribute specifying the number of heads
                    heads=heads,
                    # Attribute specifying whether the output needs to be
                    # squeezed
                    squeezed=get_by_name(merge0.attribute, "squeezed").i,
                    # Unrolled heads do not produce packed tensors
                    packed=False
                )

                # Replicate the attention operator for each head
                for i in range(heads):
                    # Start by making a full copy of the original node
                    attention = copy.deepcopy(node)
                    # Get the original shape of each input to remove the head
                    # number
                    _, seq, dim = model.get_tensor_shape(attention.input[0])
                    model.set_tensor_shape(slice0.output[i], (1, seq, dim))
                    _, seq, dim = model.get_tensor_shape(attention.input[1])
                    model.set_tensor_shape(slice1.output[i], (1, seq, dim))
                    _, seq, dim = model.get_tensor_shape(attention.input[2])
                    model.set_tensor_shape(slice2.output[i], (1, seq, dim))

                    # Connect the inputs of the replica to the output of each
                    # of the new slice operators
                    attention.input[0] = slice0.output[i]
                    attention.input[1] = slice1.output[i]
                    attention.input[2] = slice2.output[i]

                    # Get the original shape the output to remove the head
                    # number
                    _, seq, dim = model.get_tensor_shape(attention.output[0])
                    model.set_tensor_shape(merge0.input[i], (1, seq, dim))

                    # Connect the output of the attention replica to the input
                    # of the new merge operator
                    attention.output[0] = merge0.input[i]
                    # Insert the new node into the graph
                    graph.node.insert(index + i + 1, attention)
                # Insert the new slice and merge nodes into the graph
                for i, n in enumerate([slice0, slice1, slice2, merge0]):
                    # Insert the new node into the graph at index offset by
                    # number of heads
                    graph.node.insert(index + heads + i + 1, n)
                # Remove the original attention operator from the graph
                graph.node.remove(node)
        # After rewiring need to re-do the shape annotations
        # model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Script entrypoint
if __name__ == '__main__':
    # Load the model graph
    model = ModelWrapper("attention.transformed.onnx")

    # Try to infer reshaping of attention heads
    model = model.transform(InferMultiHeads())
    # Try to mode the mult-head slicing past the multi thresholds
    model = model.transform(MoveSliceMultiHeadsPastMultiThreshold())
    # Try to infer a ScaledDotProductAttention custom op
    #   Note: No further transformations can be run after this currently, as
    #   using a finn custom-op cannot be looked up for shape inference.
    model = model.transform(InferScaledDotProductAttention())
    # Parallelize attention head in the onnx graph
    model = model.transform(UnrollMultiHeadAttention())

    # Clean up the names for debugging
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    # Cleanup transformations
    from transform import Squeeze, RemoveIdentityTranspose

    # Remove dimensions of size 1 (single batch tensors)
    model = model.transform(Squeeze())
    model = model.transform(RemoveIdentityTranspose())

    # Save the inferred graph
    model.save("attention.inferred.onnx")
