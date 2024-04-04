# YAML for loading experiment configurations
import yaml
# JSON for folding configuration
import json
# Builtin math functions: gcd
import math


# Greedy factorization of the MVAU folding product to achieve the T^2 cycles per
# sample target
def factorize_mvau_folding_product(emb_dim, mlp_dim, seq_len):
    # Compute the folding product constraining the parallelization of the MVAU
    # to achieve the T^2 cycles per sample target
    mvau_folding_product = emb_dim * mlp_dim // seq_len

    # If the folding product is below 1, there is no need to parallelize at all,
    # as the model is fully dominated by the sequence length, i.e., even fully
    # sequential MVAU would still achieve the T^2 cycles per sample target
    if mvau_folding_product <= 1:
        # No parallelization
        return 1, 1

    # Checks whether n is a common divisor of both MVAU dimensions, i.e., input
    # and output
    def common_divisor(n):
        return (emb_dim % n == 0) and (mlp_dim % n == 0)

    # Common divisors of the input/output dimensions of the MVAU, these are all
    # the candidate folding configurations
    common_divisors = [
        n for n in range(1, min(emb_dim, mlp_dim)) if common_divisor(n)
    ]
    # Generate all possible SIMD-PE pairings and take the first one giving the
    # specified product. Only consider folding configurations from the set of
    # common divisors here.
    for simd in common_divisors:  # noqa
        for pe in common_divisors:  # noqa
            # Check whether this combination is a factorization of the folding
            # product, i.e., fulfills the T^2 cycles per sample folding
            # constraint
            if simd * pe == mvau_folding_product:
                # Exit here with the first solution
                return simd, pe

    # No suitable folding configuration has been found, at this point only the
    # trivial folding remains possible, if the folding product divides both
    # dimensions.
    if common_divisor(mvau_folding_product):
        # Fallback to trivial factorization if no other combinations are
        # possible
        return 1, mvau_folding_product

    # No factorization of the folding constraint with factors from the set of
    # common divisors found, not even the trivial one. The best we can do now is
    # to get as close as possible to this constraint by taking the greatest
    # common divisor.
    return 1, math.gcd(emb_dim, mlp_dim)


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
    simd, pe = factorize_mvau_folding_product(emb_dim, mlp_dim, seq_len)
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
            "SIMD": [simd, ["MVAU_hls", "MVAU_rtl"]],
            "PE": [pe, ["MVAU_hls", "MVAU_rtl"]],
            # Implement memory for FIFO buffers and MVAU weights in BRAM
            "ram_style": ["auto", [
                "StreamingFIFO_hls", "StreamingFIFO_rtl", "MVAU_hls", "MVAU_rtl"
            ]]
        },
        # Residual branches need buffering before merging them again to avoid
        # deadlock.
        **{
            # There are two residual branches per layer: One skipping the scaled
            # dot-product attention and one skipping the MLP block.
            f"AddStreams_hls_{i}": {
                # Adding two buffered branches at the input, need to buffer the
                # number of cycles of the main branch, i.e., T^2
                "inFIFODepths": 2 * [seq_len ** 2],
                # Output buffers can have default sizes
                # ...
                # Parallelize along the output dimension to achieve the T^2
                # cycles per sample target
                #   Note: Cannot process less than 1 element
                "PE": max(emb_dim // seq_len, 1)
            } for i in range(2 * num_layers)
        },
        # Residual branches contain standalone mult-thresholds which need to
        # operate in parallel
        **{
            # There are two residual branches per layer: One skipping the scaled
            # dot-product attention and one skipping the MLP block. Each has 2
            # standalone thresholds in front of the AddStreams_Batch. There is
            # another, final, standalone thresholds at the end of the model,
            # preceding the classification head.
            f"Thresholding_rtl_{i}": {
                # Parallelize along the output dimension to achieve the T^2
                # cycles per sample target
                #   Note: Cannot process less than 1 element
                "PE": max(emb_dim // seq_len, 1)
            } for i in range(2 * 4 * num_layers + 1)
        },
        # Generate a FIFO buffer and parallelization configuration for attention
        # heads
        **{
            # There are num_heads attention heads per layer of the transformer
            f"ScaledDotProductAttention_hls_{i}": {
                # Three buffered input streams to each attention head:
                #   queries, keys and values
                "inFIFODepths": 3 * [seq_len],
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
