# Protobuf onnx graph node type
from onnx import NodeProto, TensorProto
# Helper for creating ONNX nodes
from onnx import helper as oh  # noqa
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# QONNX graph transformations for inferring datatypes and shapes
from qonnx.transformation.infer_shapes import InferShapes
# Gets items from protobuf by name
from qonnx.util.basic import get_by_name
# QONNX graph transformation base class
from qonnx.transformation.base import Transformation
# for warnings
import warnings

# Collapses repeated transpose operations into a single transpose operation
# having the same effect
# Moves a transpose operator past an elementwise addition
class CollapseRepeatedTranspose(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
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
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)[0]
                # If Transpose is the final operation in the graph, there might
                # be no successor
                if successor is None or successor.op_type != "Transpose":
                    # Softly skip this node
                    continue

                # Get the (optional) permutation indices of the first transpose
                # in case it is a multi-axis transpose
                perm1 = get_by_name(node.attribute, "perm")
                # Convert permutation indices to list of integers
                perm1 = perm1.ints if perm1 is not None else None

                # Get the (optional) permutation indices of the second transpose
                # in case it is a multi-axis transpose
                perm2 = get_by_name(successor.attribute, "perm")
                # Convert permutation indices to list of integers
                perm2 = perm2.ints if perm2 is not None else None

                # Get the shape of the input tensor
                shape = model.get_tensor_shape(
                    node.input[0], fix_missing_init_shape=True
                )
                # List of dimension indices in order
                dims = (range(len(shape)))

                # Substitute the permutation indices by the reversed index list
                # of they are not given: This is default behavior, see the docs:
                #   https://onnx.ai/onnx/operators/onnx__Transpose.html
                perm1 = list(reversed(dims)) if perm1 is None else perm1
                perm2 = list(reversed(dims)) if perm2 is None else perm2

                # Combined permutation permutes the first permutation of the
                # dimensions according to the second permutation
                perm = [perm1[i] for i in perm2]

                # Create a new Transpose operator replacing the other two
                transpose = oh.make_node(
                    # Name of the operator type
                    "Transpose",
                    # Connect to the inputs to the first transpose
                    inputs=node.input,
                    # Connect to the outputs of the second transpose
                    outputs=successor.output,
                    # Insert the new permutation indices
                    perm=perm
                )
                # Insert the collapsed transpose operator
                graph.node.insert(index + 2, transpose)
                # Remove the two original transpose operators
                graph.node.remove(node)
                graph.node.remove(successor)
                # Track whether the graph has been modified, never resets to
                # False
                graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# qonnx auto merges consecutive transpose operations. This prevents some Streamling for Transposed BatchNorm later on.
# Here we try to fix this by detecting: 
# Transpose -> BatchNorm -> Something that is Transpose but differse in shape compared to the first Transpose
class RestoreTransposeAfterBatchNorm(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Transpose operation types
            if node.op_type == "Transpose":
                # Currently we dont care about joins or forks transpose only has one input
                predecessor = model.find_direct_predecessors(node)[0]
                
                # there might be no predecessor on the first node in the graph
                if predecessor is None or predecessor.op_type != "BatchNormalization":
                    # skip all non BatchNorms
                    continue
                
                # Get the permutation indices of the first transpose
                perm_this = get_by_name(node.attribute, "perm")
                # Convert permutation indices to list of integers
                perm_this = perm_this.ints if perm_this is not None else None
                if perm_this is None:
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping tranpose restore: "
                        f"no permutation found for node {predecessor_predecessor}"
                    )
                    continue

                # BatchNorm only has one input
                predecessor_predecessor = model.find_direct_predecessors(predecessor)[0]
                
                if predecessor_predecessor is None or predecessor_predecessor.op_type != "Transpose":
                    # Softly skip this node
                    continue
                
                # Get the permutation indices of the second transpose
                perm_before = get_by_name(predecessor_predecessor.attribute, "perm")
                # Convert permutation indices to list of integers
                perm_before = perm_before.ints if perm_before is not None else None
                if perm_before is None:
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping tranpose restore: "
                        f"no permutation found for node {predecessor_predecessor}"
                    )
                    continue

                if perm_before == perm_this:
                    # skip transposes that are equal already
                    continue               
                
                # apply permutation 'perm_before' to 'perm_this' and create a new array
                new_perm_this_node =  [perm_before[i] for i in perm_this]

                # new transpose to fit in between BN and new this node
                proper_name = "/".join(predecessor.name.split('/')[:-1])
                proper_name += "/restored_bn_transpose"
                restored_transpose = oh.make_node(
                    name=proper_name,
                    op_type=node.op_type, # this is also a Transpose
                    outputs=[proper_name + "_output_0"],
                    inputs=predecessor.output,
                    perm=perm_before,
                )
                graph.node.insert(index, restored_transpose)

                # replacement for this node and remove old node  
                new_this_node = oh.make_node(
                    op_type=node.op_type,
                    inputs=restored_transpose.output,
                    outputs=node.output,
                    name=node.name,
                    perm=new_perm_this_node
                )
                graph.node.remove(node)
                graph.node.insert(index+1, new_this_node)
                
                graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Follow up to RestoreTransposeAfterBatchNorm.
# Finds duplicate parralel Transposes after BatchNorm and combines them to one common fork node.
class CombineParallelTransposeAfterBatchNorm(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Applies to Transpose operation types
            if node.op_type == "BatchNormalization":
                # has to be fork node 
                if not model.is_fork_node(node):
                    continue
                # Bn only has one input
                predecessor = model.find_direct_predecessors(node)[0]
                # skip BatchNormalization where we dont have a Transpose in front
                if predecessor is None or predecessor.op_type != "Transpose":
                    continue

                 # Get the permutation indices of the first transpose
                perms = get_by_name(predecessor.attribute, "perm")
                # Convert permutation indices to list of integers
                perms = perms.ints if perms is not None else None
                if perms is None:
                    warnings.warn(
                        f"{self.__class__.__name__}: Skipping tranpose combine: "
                        f"no permutation found for node {predecessor}"
                    )

                # gather Transposes
                successors = model.find_direct_successors(node)
                # skip if there is only one successor after this
                if successors is None or len(successors) <= 1:
                    continue
                
                # search successors for Transpose and check perm shapes
                succs_to_combine = []
                for succ in successors:                    
                    if succ.op_type == "Transpose":
                        succ_perms = get_by_name(predecessor.attribute, "perm")
                        succ_perms = succ_perms.ints if succ_perms is not None else None
                        if succ_perms is None:
                            warnings.warn(
                                f"{self.__class__.__name__}: Skipping tranpose combine for successor node: "
                                f"no permutation found for node {succ}"
                            )
                        if succ_perms == perms: 
                            succs_to_combine.append(succ)
                        
                # gather successors of successors for which we need to change inputs
                next_nodes = []
                for succ in succs_to_combine:
                    next_nodes += model.find_direct_successors(succ)

                # replacement for all transpose nodes
                proper_name = "/".join(node.name.split('/')[:-1])
                proper_name += "/combined_bn_transpose"
                new_node = oh.make_node(
                    op_type="Transpose",
                    inputs=node.output,
                    outputs=[proper_name + "_output_0"],
                    name=proper_name,
                    perm=perms
                )

                # rmeove old nodes and add the new
                for succ in succs_to_combine:
                    graph.node.remove(succ)

                graph.node.insert(index+1, new_node)
                
                # set inputs for succ_succ to output of new node
                for n in next_nodes:
                    n.input[0] = new_node.output[0]
                
                graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


# Moves a transpose operator past elementwise addition or multiplication
class MoveTransposePastEltwise(Transformation):
    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
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
                # As this is not a fork-node, there can be at most one successor
                successor = model.find_direct_successors(node)[0]
                # If Transpose is the final operation in the graph, there might
                # be no successor
                if successor is None:
                    # Softly skip this node
                    continue
                # Applies to elementwise add operations
                if successor.op_type in {"Add", "Mul"}:
                    # Get names of all tensors involved in connecting the nodes
                    inp = node.input[0]
                    mid = node.output[0]
                    out = successor.output[0]

                    # y = x^T + a <=> y = (x + a^T)^T

                    # Assume left-to-right order of input to the Add operator
                    xt, a = successor.input
                    # Check whether the assumption holds true
                    if xt != mid:
                        # Leaves only the option of a and xt commuting
                        xt, a = a, xt
                    # If this assumption still does not hold true, something is
                    # wrong with the graph
                    assert xt == mid, f"Messed up graph pattern at {node.name}"

                    # Get the (optional) permutation indices of the transpose in
                    # case it is a multi-axis transpose
                    perm = get_by_name(node.attribute, "perm")
                    # Convert permutation indices to list of integers
                    perm = perm.ints if perm is not None else None

                    # This transformation does only apply to Add nodes where the
                    # second input is a constant initializer
                    if (value := model.get_initializer(a)) is not None:
                        # Transpose the initializer and re-insert into the model
                        model.set_initializer(a, value.transpose(perm))
                        # Rewire the graph to feed original input and the
                        # transposed initializer into the Add node first
                        successor.input[:] = [inp, a]
                        # Repurpose the middle tensor for the output of the
                        # addition
                        successor.output[0] = mid
                        # The Transpose operator now gets the middle tensor as
                        # its input
                        node.input[0] = mid
                        # Transpose now produces the original output tensor
                        node.output[0] = out
                        # Delete the shape annotation of the connecting tensors
                        # to be re-done later
                        model.set_tensor_shape(inp, None)
                        model.set_tensor_shape(mid, None)
                        model.set_tensor_shape(out, None)
                        # Track whether the graph has been modified, never
                        # resets to False
                        graph_modified = True
        # Need to redo the shape inference after potentially removing nodes
        model = model.transform(InferShapes())  # noqa: Shadows model
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified
