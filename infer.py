# Output warning messages
import warnings
# Standard math functions
import math
# Protobuf onnx graph node type
from onnx import NodeProto  # noqa
# Helper for creating ONNX nodes
from onnx import helper as oh  # noqa
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX datatypes
from qonnx.core.datatype import BaseDataType
# QONNX graph transformation base class
from qonnx.transformation.base import Transformation
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
                    # TODO: Near match. But what is this? just skip?
                    continue
                # There must be a Transpose feeding the key input
                transpose = model.find_producer(lhs[-1].input[1])
                # The transform applies only to transpose with exactly one input
                if transpose is None or len(transpose.input) != 1:
                    # TODO: Near match. But what is this? just skip?
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
                    "DequantSoftmax": dequant_softmax.input[0]
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
            # Head reshaping is triggered by detecting a reshape operation
            if is_reshape(node):
                # The reshape may not be a join or fork node
                if model.is_join_node(node) or model.is_fork_node(node):
                    # Softly skip this node
                    continue
                # Get the single successor node
                transpose = model.find_direct_successors(node)[0]
                # The consumer must be Transpose finalizing the reshaping
                if not is_transpose(transpose):
                    # Softly skip this node
                    continue
                # The transpose may not fork or join either
                if (model.is_join_node(transpose)
                        or model.is_fork_node(transpose)):
                    # Softly skip this node
                    continue

                # Get the input and output tensor names to the pattern
                inp = node.input[0]
                mid = node.output[0]
                end = transpose.output[0]

                # The input shape determines the sequence length, batch size
                # and embedding dimension
                seq, bsz, dim = model.get_tensor_shape(inp)

                # Currently only single-sample batches are supported
                if bsz != 1:
                    # Issue a warning of near match of the supported head
                    # pattern
                    # @formatter:off
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping near match: "
                        f"Unsupported batch size near {node.name}: {bsz}"
                    )
                    # @formatter:on
                    # Skip transforming this instance
                    continue
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
                    model.set_tensor_shape(maybe_mid,
                                           (heads, seq, dim // heads))

                # Fixed node attributes and extracted input/output/initializer
                # tensor names
                kwargs = {
                    # Refer to this operator type by its name
                    "op_type": "SliceMultiHeads",
                    # Execution will try to look up the implementation in the
                    # package referred to by the domain
                    "domain": "finn.custom_op.fpgadataflow",
                    # Execution backend: Required attribute inherited from
                    # HLSCustomOp
                    "backend": "fpgadataflow",
                    # Named inputs extracted from the graph pattern
                    "inputs": [inp],
                    # Named outputs extracted from the graph pattern
                    "outputs": [maybe_mid],
                    # Give node name derived from the operator type and the name
                    # of the triggering node to be removed
                    "name": f"MultiHeads_{node.name}"
                }

                # Extract the node attributes of the multi heads operator from
                # all constituent nodes
                node_attrs = {
                    "heads": heads
                }

                # Create a new custom node replacing the multi head reshape
                heads = oh.make_node(**kwargs, **node_attrs)
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

        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Script entrypoint
if __name__ == '__main__':
    # Load the model graph
    model = ModelWrapper("attention.transformed.onnx")

    # # Try to infer reshaping of attention heads
    # model = model.transform(InferMultiHeads())

    # Try to infer a ScaledDotProductAttention custom op
    model = model.transform(InferScaledDotProductAttention())

    # Save the inferred graph
    model.save("attention.inferred.onnx")
