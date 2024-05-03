# Use the DVC api for loading the YAML parameters
import dvc.api
# JSON for folding configuration
import json

# Seeding RNGs for reproducibility
from utils import seed


# Generates a folding configuration for a transformer model of multiple layers
# each with multiple heads
def make_folding(num_heads, num_layers, seq_len, **_):
    # Maximum number of inputs/outputs expected for any operator in the model.
    # For a single head dummy, this is 4 corresponding to branching of residual
    # skip connection plut queries, keys and values input of the single
    # attention head otherwise this is determined by splitting to the number of
    # attention heads.
    max_inputs_outputs_per_op = max(4, num_heads)
    # Start filling the folding configuration with defaults applied to all
    # operators
    folding = {  # noqa: Shadows outer scope
        "Defaults": {
            # Set all input FIFO depths to 2 to avoid out-of-bounds access when
            # FINN tries to instantiate FIFOs at the inputs of attention head
            # merging. Depths 2 FIFOs will be removed afterward, thus this has
            # no real effect on the generated hardware.
            "inFIFODepths": [max_inputs_outputs_per_op * [2], "all"],
            # Set all output FIFO depths to 2, same argument as above.
            "outFIFODepths": [max_inputs_outputs_per_op * [2], "all"],
            # Let the tools automatically decide how to implement memory for
            # FIFO buffers and MVAU weights
            "ram_style": ["auto", [
                "StreamingFIFO_hls", "StreamingFIFO_rtl", "MVAU_hls", "MVAU_rtl"
            ]]
        },
        # Residual branches need buffering before merging them again to avoid
        # deadlock.
        **{
            # There are two residual branches per layer: One skipping the scaled
            # dot-product attention and one skipping the MLP block. There is
            # also one positional encoding at the input.
            f"ElementwiseAdd_hls_{i}": {
                # Adding two buffered branches at the input, need to buffer the
                # number of cycles of the main branch, i.e., T^2
                # Adding the positional encoding needs to be parallelized but
                # does not require extra buffering (default to depth 2 FIFOs).
                "inFIFODepths": 2 * [seq_len ** 2 if i != 0 else 2],
            } for i in range(2 * num_layers + 1)
        },
        # Generate a FIFO buffer and parallelization configuration for attention
        # heads
        **{
            # There are num_heads attention heads per layer of the transformer
            f"ScaledDotProductAttention_hls_{i}": {
                # Three buffered input streams to each attention head:
                #   queries, keys and values
                "inFIFODepths": 3 * [seq_len],
            } for i in range(num_layers * num_heads)
        }
    }
    # Return the generated folding configuration dictionary
    return folding


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml")
    # Seed all RNGs
    seed(params["seed"])
    # Generate a folding config for the model configuration
    folding = make_folding(**params["model"])
    # Dump the folding dictionary as json
    with open(params["build"]["folding_config_file"], "w") as file:
        # Format dictionary as JSON and ump into file
        json.dump(folding, file)
