# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Base class for all QONNX graph transformations
from qonnx.transformation.general import Transformation
# Resolves the QONNX CustomOp corresponding to the plain ONNX node
from qonnx.custom_op.registry import getCustomOp


# Applies configuration dictionary to the model graph
class ApplyConfig(Transformation):
    # Initializes the transformation with the configuration dictionary
    def __init__(self, config):
        # Initialize the transformation base class
        super().__init__()
        # Register the configuration dictionary to be used in apply()
        self.config = config

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # A node should not be named "defaults"...
            assert node.name != "defaults", \
                "Node has reserved name 'defaults'"
            # Convert this to the custom-op instance for easy access to node
            # attributes
            inst = getCustomOp(node)
            # Apply the per operator type default configurations to the node
            if node.op_type in self.config["defaults"]:
                # Run over all default options to be applied to this node
                for key, value in self.config["defaults"][node.op_type].items():
                    # Set the nodes attribute to the default option value
                    inst.set_nodeattr(key, value)
            # If there is an individual, node-specific configuration apply
            # this next, potentially overriding the defaults set above
            if node.name in self.config:
                # Run over all node-specific options to be applied to this
                # node
                for key, value in self.config[node.name].items():
                    # Set the nodes attribute to the option value
                    inst.set_nodeattr(key, value)
        # Return model with configuration applied
        # Note: Do not consider this as modifying the graph. This does not have
        # to be reapplied multiple times.
        return model, False
