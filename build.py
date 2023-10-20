# FINN dataflow builder
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import AutoFIFOSizingMethod
# QONNX wrapper of ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Detects the attention pattern and converts to HLS custom op
from infer import InferScaledDotProductAttention


# Function running the InferScaledDotProductAttention transformation
def step_convert_attention_to_hls(model: ModelWrapper, _):
    # Applies just the single transformation
    return model.transform(InferScaledDotProductAttention())


# Create a configuration for building the scaled dot-product attention operator
# to a hardware accelerator
cfg = build_cfg.DataflowBuildConfig(
    # Where to write the outputs
    output_dir="attention-build",
    # Try to synthesize for 100 MHz clock, i.e., 10 ns
    synth_clk_period_ns=10.0,
    # Build for the Pynq-Z1 board
    board="Pynq-Z1",
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
    # Save the intermediate model graphs
    save_intermediate_models=True,
    # Avoid RTL simulation for setting the FIFO sizes
    auto_fifo_strategy=AutoFIFOSizingMethod.CHARACTERIZE,
    # Build steps to execute
    steps=[
        "step_qonnx_to_finn",
        "step_tidy_up",
        "step_streamline",
        # New conversion of the scaled dot-product attention pattern
        step_convert_attention_to_hls,
        "step_convert_to_hls",
        "step_create_dataflow_partition",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        # The ScaledDotProductAttention custom op does not define any estimates
        # "step_generate_estimate_reports",
        "step_hls_codegen",
        "step_hls_ipgen",
        # Attention RTL sim is not implemented due to missing float IPs. All
        # following steps rely in some way on RTL simulation and thus cannot be
        # executed currently.
        # "step_set_fifo_depths",
        # "step_create_stitched_ip",
        # "step_measure_rtlsim_performance",
        # "step_out_of_context_synthesis",
        # "step_synthesize_bitfile",
        # "step_make_pynq_driver",
        # "step_deployment_package",
    ]
)
# Run the build process on the dummy attention operator graph
build.build_dataflow_cfg("attention.onnx", cfg)
