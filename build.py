# FINN dataflow builder
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import AutoFIFOSizingMethod

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

# Remove some operations without real effect
from transformation.remove import RemoveIdentityTranspose, RemoveIdentityReshape
# Cleanup transformations
from transformation.squeeze import Squeeze
# Detects the attention pattern and converts to HLS custom op
from transformation.attention import InferScaledDotProductAttention
# Mult-Head Attention support
from transformation.attention_heads import (
    InferMultiHeads,
    MoveSplitMultiHeadsPastMultiThreshold,
    UnrollMultiHeadAttention
)


# Function running transformations necessary to clean up models containing
# attention operators
def step_tidy_up_pre_attention(model: ModelWrapper, _):
    # Add shape and datatype annotations throughout all the graph
    model = model.transform(InferDataTypes())  # noqa Duplicate
    model = model.transform(InferShapes())

    # Cleanup the graph by removing redundant, unnecessary and constant nodes
    # and tensors and give unique names to everything remaining
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(FoldConstants())

    # Remove unnecessary shape and layout transformations
    model = model.transform(RemoveIdentityReshape())
    model = model.transform(RemoveIdentityTranspose())
    # Insert tensor layout annotations for Quant tot MultiThreshold transform
    # to determine the correct output channel dimension
    model = model.transform(InferDataLayouts())
    # Return the tidied up model
    return model


# Variant of streamlining transformations adapted to attention operators
def step_streamline_attention(model: ModelWrapper, _):
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
    # Return the streamlined model
    return model


# Function running the InferScaledDotProductAttention transformation
def step_convert_attention_to_hls(model: ModelWrapper, _):
    # Try to infer reshaping of attention heads
    model = model.transform(InferMultiHeads())
    # Try to mode the mult-head splitting past the multi thresholds
    model = model.transform(MoveSplitMultiHeadsPastMultiThreshold())
    # Try to infer a ScaledDotProductAttention custom op
    #   Note: No further transformations can be run after this currently, as
    #   using a finn custom-op cannot be looked up for shape inference.
    model = model.transform(InferScaledDotProductAttention())
    # Parallelize attention head in the onnx graph
    model = model.transform(UnrollMultiHeadAttention())
    # Return the model with attention and multi-heads mapped to hls operators
    return model


# Post-processing tidy-up squeezing dimensions and identity operators left over
# from mapping the attention operators
def step_tidy_up_post_attention(model: ModelWrapper, _):
    # Remove dimensions of size 1 (single batch tensors)
    model = model.transform(Squeeze())
    model = model.transform(RemoveIdentityTranspose())
    # Clean up the names for debugging
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    # Return the tidied up model
    return model


# Create a configuration for building the scaled dot-product attention operator
# to a hardware accelerator
cfg = build_cfg.DataflowBuildConfig(
    # Where to write the outputs
    output_dir="attention-build",
    # Try to synthesize for 100 MHz clock, i.e., 10 ns
    synth_clk_period_ns=10.0,
    # Build for the RFSoC 4x2 board
    board="RFSoC2x2",
    # This is a Zynq flow
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    # Generate and keep the intermediate outputs including reports
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
    # Steps after which verification should be run
    verify_steps=[
        # Verify the model after generating C++ HLS and applying folding
        build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM
        # No RTL Simulation support for now
    ],
    # File with test inputs for verification
    verify_input_npy="inp.npy",
    # File with expected test outputs for verification
    verify_expected_output_npy="out.npy",
    # Save the intermediate model graphs
    save_intermediate_models=True,
    # Avoid RTL simulation for setting the FIFO sizes
    auto_fifo_strategy=AutoFIFOSizingMethod.CHARACTERIZE,
    # Do not automatically set FIFO sizes as this requires RTL simulation not
    # implemented for the attention operator
    auto_fifo_depths=False,
    # Build steps to execute
    steps=[
        # Need to apply some tidy-up transformations before converting to the
        # finn dialect of onnx
        step_tidy_up_pre_attention,
        "step_qonnx_to_finn",
        "step_tidy_up",
        # Custom streamlining for models containing attention operators
        step_streamline_attention,
        # New conversion of the scaled dot-product attention pattern
        step_convert_attention_to_hls,
        # Another tidy-up step to remove unnecessary dimensions and operations
        step_tidy_up_post_attention,
        "step_tidy_up",
        "step_convert_to_hls",
        "step_create_dataflow_partition",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        # The ScaledDotProductAttention custom op does not define any estimates
        "step_generate_estimate_reports",
        "step_hls_codegen",
        "step_hls_ipgen",
        # Attention RTL sim is not implemented due to missing float IPs. All
        # following steps rely in some way on RTL simulation and thus cannot be
        # executed currently.
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        # "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile",
        "step_make_pynq_driver",
        "step_deployment_package",
    ]
)
# Run the build process on the dummy attention operator graph
build.build_dataflow_cfg("attention.onnx", cfg)
