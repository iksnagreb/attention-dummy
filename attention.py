# YAML for loading experiment configurations
import yaml

# For saving numpy array data
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas quantized variants of PyTorch layers
from brevitas.nn import (
    QuantIdentity,
    QuantMultiheadAttention,
    QuantLinear,
    QuantReLU,
    QuantEltwiseAdd
)
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
            self,
            num_heads,
            num_layers,
            bias,
            emb_dim,
            mlp_dim,
            seq_len,
            bits,
            mask
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

        # Input quantizer preceding the first attention block
        self.input_quant = QuantIdentity(
            # Note: Quantize inputs to 8 bit to make verification pass. No idea
            # why it fails otherwise, probably precision and/or rounding issues.
            act_quant=Int8ActPerTensorFloat, return_quant_tensor=True
        )
        # Quantized multi-head attention operator from brevitas
        self.attentions = torch.nn.ModuleList([
            QuantMultiheadAttention(
                # Size of the embedding dimension (input and output)
                embed_dim=emb_dim,
                # Number of attention heads
                num_heads=num_heads,
                # Enable a bias added to the input and output projections
                bias=bias,
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
                in_proj_input_quant=None,

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
                out_proj_output_quant=None,

                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True
            ) for _ in range(num_layers)
        ])
        # Point-wise mlp following the attention block made up of two quantized
        # linear layers with ReLU non-linear activation in between and afterward
        self.mlps = torch.nn.ModuleList([torch.nn.Sequential(
            # First mlp layer projecting to the mlp dimension
            QuantLinear(
                # Inputs have the size of the attention embedding dimension
                emb_dim,
                # Project to the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Enable the learned bias vector
                bias=bias,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=WeightQuantizer,
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=BiasQuantizer,
                # Quantize the input of the layer
                input_quant=None,
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True
            ),
            # Use the ReLU activation function instead of the more commonly used
            # GELU, as the latter is not mapped easily to hardware with FINN
            QuantReLU(
                # Note: ReLU must be quantized to unsigned representation
                act_quant=UnsignedActQuantizer,
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True
            ),
            # Second mlp layer projecting back to the embedding dimension
            QuantLinear(
                # Inputs have the configured mlp dimension, which is typically
                # larger than the embedding dimension
                mlp_dim,
                # Project back to the size of the attention embedding dimension
                emb_dim,
                # Enable the learned bias vector
                bias=bias,
                # Quantize weights to the same representation as all other
                # layers
                weight_quant=WeightQuantizer,
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=BiasQuantizer,
                # No input quantizer as the inputs are already quantized by the
                # preceding ReLU layer
                input_quant=None,
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            )
        ) for _ in range(num_layers)])

        # Quantizer for the first residual branch per layer, i.e., the one
        # skipping past the attention block
        self.residuals1 = torch.nn.ModuleList([
            # Use a quantized elementwise add with shared input quantizer but
            # no output quantizer
            QuantEltwiseAdd(
                # Shared input activation quantizer such that the scales at both
                # input branches are identical. This allows floating point scale
                # factor to be streamlined past the add-node.
                input_quant=ActQuantizer,
                # Disable the output quantizer after the add operation for now
                # TODO: What do we actually need here?
                output_quant=None,
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            ) for _ in range(num_layers)
        ])

        # Quantizer for the second residual branch per layer, i.e., the one
        # skipping past the MLP block
        self.residuals2 = torch.nn.ModuleList([
            # Use a quantized elementwise add with shared input quantizer but
            # no output quantizer
            QuantEltwiseAdd(
                # Shared input activation quantizer such that the scales at both
                # input branches are identical. This allows floating point scale
                # factor to be streamlined past the add-node.
                input_quant=ActQuantizer,
                # Disable the output quantizer after the add operation for now
                # TODO: What do we actually need here?
                output_quant=None,
                # Pass quantization information on to the next layer.
                # Note: Not for the last layer to allow this to be combined with
                # standard pytorch calls like .detach() or .numpy(), which are
                # not directly available on QuantTensor.
                return_quant_tensor=(layer != (num_layers - 1))
            ) for layer in range(num_layers)
        ])

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
        # Move the mask to the same device as the input, just in case...
        mask = self.mask.to(x.device) if self.mask is not None else None
        # Quantize the input preceding the first attention layer
        x = self.input_quant(x)  # noqa: Shadows name 'x' from outer scope
        # Iterate over all layers from the four module lists zipped together
        layers = zip(
            self.attentions, self.mlps, self.residuals1, self.residuals2
        )
        # There are multiple blocks of attention
        for attention, mlp, residual1, residual2 in layers:
            # TODO: Here would be the first pre-layer-normalization
            # Masked self-attention operation with residual connection
            x = residual1(x, attention(x, x, x, attn_mask=mask)[0]) # noqa:
            # Shadows name 'x' from outer scope
            # TODO: Here would be the second pre-layer-normalization
            # Point-wise fully connected MLP block with residual connection
            x = residual2(x, mlp(x))  # noqa: Shadows name 'x' from outer scope
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
    # Pass random data through the model to "calibrate" dummy quantizer. Large
    # batch to have more calibration samples. Otherwise, there is too much
    # deviation between this calibration and the verification samples.
    model(torch.rand(32786, seq, dim))
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
