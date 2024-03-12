# YAML for loading experiment configurations
import yaml
# JSON for folding configuration
import json


# Greedy factorization of the MVAU folding product to achieve the T^2 cycles per
# sample target
def factorize_mvau_folding_product(mvau_folding_product):
    # Parallelization/folding cannot be negative or zero
    if mvau_folding_product < 1:
        # Trivial folding/parallelization: No parallelization
        return 1, 1
    # Generate all possible SIMD-PE pairings and take the first one giving the
    # specified product. Note that range is exclusive at the upper bound, this
    # avoids the trivial factorization (1, mvau_folding_product)
    for simd in range(mvau_folding_product):  # noqa
        for pe in range(mvau_folding_product):  # noqa
            # Check whether this combination is a factorization of the folding
            # product, i.e., fulfills the T^2 cycles per sample folding
            # constraint
            if simd * pe == mvau_folding_product:
                # Exit here with the first solution
                return simd, pe
    # Fallback to trivial factorization if no other combinations are possible
    return 1, mvau_folding_product


# Generates a folding configuration for a transformer model of multiple layers
# each with multiple heads
def make_folding(num_heads, num_layers, emb_dim, mlp_dim, seq_len, **_):
    # Maximum number of inputs/outputs expected for any operator in the model.
    # For a single head dummy, this is 4 corresponding to branching of residual
    # skip connection plut queries, keys and values input of the single
    # attention head otherwise this is determined by splitting to the number of
    # attention heads.
    max_inputs_outputs_per_op = max(4, num_heads)
    # For achieving the T^2 cycles per sample target, parallelization of
    # all MVAUs must fulfill the following constraint:
    #   SIMD * PE = emb_dim * mlp_dim / seq_len
    simd, pe = factorize_mvau_folding_product(emb_dim * mlp_dim // seq_len)
    # All operators which really need FIFO buffers are determined by the
    # attention heads, which require buffering of the whole sequence per head.
    fifo_depths = round(seq_len * emb_dim / num_heads)
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
            # Set the parallelism of all MVAUs to meet the T^2 cycles per sample
            # target as computed above.
            "SIMD": [simd, ["MatrixVectorActivation"]],
            "PE": [pe, ["MatrixVectorActivation"]],
            # Implement memory for FIFO buffers and MVAU weights in BRAM
            "ram_style": ["block", ["StreamingFIFO", "MatrixVectorActivation"]]
        },
        # Residual branches need buffering before merging them again to avoid
        # deadlock.
        **{
            # There are two residual branches per layer: One skipping the scaled
            # dot-product attention and one skipping the MLP block.
            f"AddStreams_Batch_{i}": {
                # Adding two buffered branches at the input
                "inFIFODepths": 2 * [fifo_depths]
                # Output buffers can have default sizes
                # ...
            } for i in range(2 * num_layers)
        },
        # Generate a FIFO buffer and parallelization configuration for attention
        # heads
        **{
            # There are num_heads attention heads per layer of the transformer
            f"ScaledDotProductAttention_{i}": {
                # Three buffered input streams to each attention head:
                #   queries, keys and values
                "inFIFODepths": 3 * [fifo_depths],
                # Output buffers can have default sizes
                # ...
                # No parallelization along the sequence axis,
                #   i.e., process seq_len groups of input in sequence
                "SeqFold": seq_len,
                # Full parallelization along the embedding axis,
                #   i.e., process 1 group of input elements in parallel
                "EmbFold": 1
            } for i in range(num_layers * num_heads)
        }
    }
    # Return the generated folding configuration dictionary
    return folding


# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)
    # Generate a folding config for the model configuration
    folding = make_folding(**params["model"])
    # Dump the folding dictionary as json
    with open(params["build"]["folding_config_file"], "w") as file:
        # Format dictionary as JSON and ump into file
        json.dump(folding, file)
