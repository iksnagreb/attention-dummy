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
# Convert ONNX nodes to QONNX custom ops
from qonnx.custom_op.registry import getCustomOp
# QONNX graph transformation base class
from qonnx.transformation.base import Transformation
# Transformation running onnx shape inference
from qonnx.transformation.infer_shapes import InferShapes
# Utility function for transforming ONNX graphs
from transformation.util import (
    op_types,
    is_mul,
    is_matmul,
    is_join_matmul,
    is_softmax,
    all_upstream_to_matmul
)


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

                # If there is no dequant softmax yet, check alternative pattern
                if dequant_softmax is None:
                    # Alternatively, there might not be a quantizer following
                    # the softmax
                    dequant_softmax = lhs[1] if is_softmax(lhs[0]) else None

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

                # If there is a dequant scale factor, try to lift it from
                # initializer to node attribute
                if dequant_softmax is not None:
                    # Get the initializer tensor
                    scale = model.get_initializer(dequant_softmax.input[1])
                    # This must be an initializer, the attention operator
                    # currently does not handle any dynamically produced scale
                    # factors
                    if scale is None:
                        # Issue a warning of near match of the supported
                        # attention pattern
                        # @formatter:off
                        warnings.warn(
                            f"{self.__class__.__name__}: Skipping near match: "
                            f"Non-constant dequantizer near {node.name}: "
                            f" {dequant_softmax.name}"
                        )
                        # @formatter:on
                        # Skip transforming this instance
                        continue
                    # Currently, only scalar dequantizer scale factors are
                    # supported
                    if not all(x == 1 for x in scale.shape):
                        # Issue a warning of near match of the supported
                        # attention pattern
                        # @formatter:off
                        warnings.warn(
                            f"{self.__class__.__name__}: Skipping near match: "
                            f"Non-scalar dequantizer near {node.name}: "
                            f" {dequant_softmax.name}"
                        )
                        # @formatter:on
                        # Skip transforming this instance
                        continue
                    # Extract the single float value of the tensor
                    dequant_softmax = float(scale.item())
                # Insert default scale if the is no dequantizer present
                else:
                    # Default is identity scale
                    dequant_softmax = 1.0

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

                # Output type of the first matmul
                out_qk_matmul = lhs[-1].output[0]
                # Extend the output type to include the optional thresholding
                # activation
                if act_qk_matmul is not None:
                    # Single output tensor of the activation function
                    out_qk_matmul = act_qk_matmul.output[0]

                # Extract output bias of the thresholding activation functions
                def out_bias(act):
                    # Does only apply to thresholding activations
                    if act is not None and act.op_type == "MultiThreshold":
                        # Extract via interpreting the node a sQONNX custom op
                        return getCustomOp(act).get_nodeattr("out_bias")
                    # Default bias if no bias
                    return 0.0

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
                    "inputs": [q, k, v, *thresholds],
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
                    "OutQKMatMul": model.get_tensor_datatype(out_qk_matmul),
                    # Activation function type following the first matmul
                    "ActQKMatMul": act_op_type_str(act_qk_matmul),
                    # Output bias to be applied to the thresholding activation
                    # following the Query x Key multiplication
                    "BiasActQKMatMul": out_bias(act_qk_matmul),

                    # Datatype of accumulator elements of the second matmul
                    "AccAVMatMul": model.get_tensor_datatype(node.output[0]),
                    # Datatype of output elements of the second matmul
                    # Note: Always the same as the OType
                    "OutAVMatMul": model.get_tensor_datatype(last.output[0]),
                    # Activation function type following the second matmul
                    "ActAVMatMul": act_op_type_str(act_av_matmul),
                    # Output bias to be applied to the thresholding activation
                    # following the Attention x Value multiplication
                    "BiasActAVMatMul": out_bias(act_av_matmul),

                    # Softmax may be preceded by a de-quantizer scalar
                    # multiplication
                    "DequantSoftmax": dequant_softmax,
                    # Datatype of softmax normalization before applying
                    # activation or type cast. This is called Acc to stick to
                    # the naming scheme of the MatMul operators before.
                    #   Note: Currently this is ALWAYS floats
                    "AccASoftmax": "FLOAT32",
                    # Activation function type following the softmax
                    # normalization of the attention weights
                    "ActASoftmax": act_op_type_str(act_a_softmax),
                    # Output bias to be applied to the thresholding activation
                    # following the softmax normalization of the attention
                    # weights
                    "BiasActASoftmax": out_bias(act_a_softmax),
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
        # After rewiring need to re-do the shape annotations
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified
