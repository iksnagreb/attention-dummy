# YAML for loading experiment configurations
import yaml

# For saving numpy array data
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas quantized variants of PyTorch layers
from brevitas.nn import QuantMultiheadAttention, QuantLinear, QuantReLU
# Brevitas to QONNX model export
from brevitas.export import export_qonnx
# Brevitas quantizer
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Uint8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int8Bias,
)


# Generates a causal attention mask
def causal_mask(length):
    return torch.nn.Transformer.generate_square_subsequent_mask(length)


# Generates a random attention mask
def random_mask(length):
    return torch.where(  # noqa
        torch.rand(length, length) > 0.5, -torch.inf, 0.0  # noqa
    )


# Minimal example of a model which could be considered as a transformer
class DummyTransformer(torch.nn.Module):
    # Initializes and registers the module parameters and state
    def __init__(
            self, num_heads, num_layers, emb_dim, mlp_dim, seq_len, bits, mask
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Derive a custom quantizer for weights based on the 8-bit variant
        class WeightQuantizer(Int8WeightPerTensorFloat):
            bit_width = bits

        # Derive a custom quantizer for the bias based on the 8-bit variant
        class BiasQuantizer(Int8Bias):
            bit_width = bits

        # Derive a custom quantizer for activations based on the 8-bit variant
        class ActQuantizer(Int8ActPerTensorFloat):
            bit_width = bits

        # Derive a custom quantizer for activations based on the 8-bit variant
        class UnsignedActQuantizer(Uint8ActPerTensorFloat):
            bit_width = bits

        # Quantized multi-head attention operator from brevitas
        self.attention_blocks = torch.nn.ModuleList(num_layers * [
            QuantMultiheadAttention(
                # Size of the embedding dimension (input and output)
                embed_dim=emb_dim,
                # Number of attention heads
                num_heads=num_heads,
                # Enable a bias added to the input and output projections
                bias=True,
                # Layout of the inputs:
                #   Batch x Sequence x Embedding (batch-first, True)
                #   Sequence x Batch x Embedding (batch-second, False)
                batch_first=True,
                # If query, key and value input are the same, packed input
                # projections use a single, large linear projection to produce
                # the actual query, key and value inputs. Otherwise, use
                # separate linear projections on each individual input.
                packed_in_proj=False,
                # Brevitas has this as an unsigned quantizer by default, but
                # finn can only handle signed quantizer
                attn_output_weights_quant=ActQuantizer,
                # Insert an additional quantizer in front ot the softmax. In our
                # finn custom-op, this will be matched to the quantizer
                # following the query and key matmul.
                softmax_input_quant=None,
                # Quantize the input projections weights to 4 bits, brevitas
                # defaults to 8 bits
                in_proj_weight_quant=WeightQuantizer,
                # Quantize the bias of the input projections to 4 bits as well
                in_proj_bias_quant=BiasQuantizer,
                # Use 4-bit inputs to the attention block
                in_proj_input_quant=ActQuantizer,

                # Quantize the output projections weights to 4 bits, brevitas
                # defaults to 8 bits
                out_proj_weight_quant=WeightQuantizer,
                # Quantize the bias of the output projections to 4 bits as well
                out_proj_bias_quant=BiasQuantizer,
                # Use 4-bit inputs to the attention block
                out_proj_input_quant=ActQuantizer,

                # Quantizer the key after projections to 4-bit activations
                k_transposed_quant=ActQuantizer,
                # Quantize the queries after projections to 4-bit activations
                q_scaled_quant=ActQuantizer,
                # Quantize the values after projection to 4-bit activations
                v_quant=ActQuantizer,

                # No output quantization for now, as stacking multiple layers
                # results in multiple multi-thresholds in succession
                out_proj_output_quant=None
            )
        ])
        # Point-wise mlp following the attention block made up of two quantized
        # linear layers with ReLU non-linear activation in between and afterward
        self.mlp_blocks = torch.nn.ModuleList(num_layers * [torch.nn.Sequential(
            # First mlp layer projecting to the mlp dimension
            QuantLinear(
                # Inputs have the size of the attention embedding dimension
                emb_dim,
                # Project to the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Enable the learned bias vector
                bias=True,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=WeightQuantizer,
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=BiasQuantizer,
                # Quantize the input of the layer
                input_quant=ActQuantizer
            ),
            # Use the ReLU activation function instead of the more commonly used
            # GELU, as the latter is not mapped easily to hardware with FINN
            #   Note: ReLU must be quantized to unsigned representation
            QuantReLU(act_quant=UnsignedActQuantizer, input_quant=ActQuantizer),
            # Second mlp layer projecting back to the embedding dimension
            QuantLinear(
                # Inputs have the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Project back to the size of the attention embedding dimension
                emb_dim,
                # Enable the learned bias vector
                bias=True,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=WeightQuantizer,
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=BiasQuantizer,
                # Quantize the input of the layer
                input_quant=ActQuantizer
            ),
            # Use the ReLU activation function instead of the more commonly used
            # GELU, as the latter is not mapped easily to hardware with FINN
            #   Note: ReLU must be quantized to unsigned representation
            QuantReLU(act_quant=UnsignedActQuantizer, input_quant=ActQuantizer)
        )])

        # Generate an attention mask depending on configuration
        #   Note: Prepare all and select from dictionary by config option
        self.mask = {
            "none": None,
            "causal": causal_mask(seq_len),
            "const": random_mask(seq_len)
        }[mask]

    # Model forward pass doing self attention, i.e, distributing a single input
    # to the query, key and value inputs of the attention operator
    def forward(self, x):  # noqa: Shadows name 'x' from outer scope
        # There are multiple blocks of attention
        for attention, mlp in zip(self.attention_blocks, self.mlp_blocks):
            # Distribute input to all three attention inputs and use output as
            # next blocks input
            x = mlp(attention(x, x, x, attn_mask=self.mask)[0])  # noqa:
            # Shadows name 'x' from outer scope
        # Return the output of the final block as the global output
        return x


# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)["model"]
    # Create a model instance from the configuration parameters
    model = DummyTransformer(**params)
    # Get the configured sequence length and embedding dimension to generate
    # test inputs
    seq, dim = params["seq_len"], params["emb_dim"]
    # First pass of random data through the model to "calibrate" dummy quantizer
    model(torch.rand(1, seq, dim))
    # Switch model to evaluation mode to have it fixed for export
    model = model.eval()
    # Sample random input tensor in batch-first layout
    x = torch.rand(1, seq, dim)
    # Compute attention output
    o = model(x)
    # Save the input and output data for verification purposes later
    np.save("inp.npy", x.detach().numpy())
    np.save("out.npy", o.detach().numpy())
    # Export the model graph to QONNX
    export_qonnx(model, (x,), "attention.onnx")
