# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas wrapper around PyTorch tensors adding quantization information
from brevitas.quant_tensor import QuantTensor
# Brevitas: Quantized versions of PyTorch layers
from brevitas.nn import (
    QuantMultiheadAttention,
    QuantEltwiseAdd,
    QuantIdentity,
    QuantLinear,
    QuantReLU
)


# Derives a weight quantizer from the brevitas bases leaving bit-width and
# signedness configurable
def weight_quantizer(bits, _signed=True):
    # Brevitas quantizer base classes
    from brevitas.quant.base import NarrowIntQuant, MaxStatsScaling
    from brevitas.quant.solver import WeightQuantSolver
    from brevitas.inject.enum import RestrictValueType

    # Derive a Quantizer from the brevitas bases
    class Quantizer(NarrowIntQuant, MaxStatsScaling, WeightQuantSolver):
        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Per tensor quantization, not per channel
        scaling_per_output_channel = False
        # What is this? Copied from PerTensorFloatScaling*
        #   Probably restricts the scale to be floating-point?
        restrict_scaling_type = RestrictValueType.FP

    # Return the derived quantizer configuration
    return Quantizer


# Derives a bias quantizer from the brevitas bases leaving bit-width and
# signedness configurable
def bias_quantizer(bits, _signed=True):
    # Brevitas quantizer base classes
    from brevitas.quant import IntBias

    # Derive a Quantizer from the brevitas bases
    class Quantizer(IntBias):
        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Do not require the bit-width to be adjusted to fit the accumulator to
        # which the bias is added
        requires_input_bit_width = False

    # Return the derived quantizer configuration
    return Quantizer


# Derives an activation quantizer from the brevitas bases leaving bit-width and
# signedness configurable
def act_quantizer(bits, _signed=True):
    # Brevitas quantizer base classes
    from brevitas.quant.base import IntQuant, ParamFromRuntimePercentileScaling
    from brevitas.quant.solver import ActQuantSolver
    from brevitas.inject.enum import RestrictValueType

    # Derive a Quantizer from the brevitas bases
    class Quantizer(
        IntQuant, ParamFromRuntimePercentileScaling, ActQuantSolver
    ):
        # Configure the quantization bit-width
        bit_width = bits
        # Signedness of the quantization output
        signed = _signed
        # Per tensor quantization, not per channel
        scaling_per_output_channel = False
        # What is this? Copied from PerTensorFloatScaling*
        #   Probably restricts the scale to be floating-point?
        restrict_scaling_type = RestrictValueType.FP

    # Return the derived quantizer configuration
    return Quantizer


# Gets the normalization layer from configuration key
def get_norm(key, normalized_shape):
    # Transposes Sequence and Embedding dimensions
    class Transpose(torch.nn.Module):
        # Forward pass transposing the feature map
        def forward(self, x):  # noqa: May be static
            # Transpose the last two dimensions of batch x seq x emb layout
            return torch.transpose(x, dim0=-1, dim1=-2)

    # Dictionary mapping keys to supported normalization layer implementations
    norms = {
        # PyTorch default layer normalization. Needs to know the shape of the
        # feature map to be normalized
        "layer-norm": torch.nn.LayerNorm(
            # Note: Disable affine parameters as potential negative scale causes
            # streamlining issues later
            normalized_shape=normalized_shape, elementwise_affine=False
        ),
        # PyTorch default 1-dimensional batch normalization. Needs to transpose
        # embedding and sequence dimension to normalized over the embedding
        # dimension, which is expected to be second.
        "batch-norm": torch.nn.Sequential(
            # Note: Disable affine parameters as potential negative scale causes
            # streamlining issues later
            Transpose(), torch.nn.LazyBatchNorm1d(affine=False), Transpose()
        ),
        # No normalization by a PyTorch built-in identity layer. Should not
        # appear in the graph.
        "none": torch.nn.Identity()
    }

    # Select the normalization layer by key
    return norms[key]


# Gets the attention mask from configuration key and shape
def get_mask(key, length):
    # Dictionary mapping keys to supported normalization layer implementations
    masks = {
        # No attention mask
        "none": None,
        # Generate the upper triangular mask for causal attention
        "causal": torch.nn.Transformer.generate_square_subsequent_mask(length),
        # Square matrix with entries randomly set to -inf or 0.0 with 50%
        # probability each
        "random": torch.where(  # noqa: Confused by types?
            torch.rand(length, length) > 0.5, -torch.inf, 0.0
        )
    }
    # Select the mask type by key
    return masks[key]


# Single-layer scaled dot-product attention block with MLP and normalization
class TransformerBlock(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(
            self, num_heads, emb_dim, mlp_dim, seq_len, bias, norm, mask, bits
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Input quantizer to the scaled dot-product attention operations, shared
        # by queries, keys and values inputs. It is important to have this
        # quantizer separate and not preceding the fork node of the residual
        # branches to avoid consecutive quantizers in the skip branch.
        # Note: For some reason it seems not to be possible to use the
        #   in_proj_input_quant of the attention operator
        self.sdp_input_quant = QuantIdentity(
            # Quantize at the output
            act_quant=act_quantizer(bits, _signed=True),
            # Pass quantization information on to the next layer.
            return_quant_tensor=True
        )
        # Quantized scaled dot-product attention operator
        self.sdp = QuantMultiheadAttention(
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
            attn_output_weights_quant=act_quantizer(bits, _signed=True),
            # Insert an additional quantizer in front ot the softmax. In our
            # finn custom-op, this will be matched to the quantizer
            # following the query and key matmul.
            # Note: Disable to prevent the quantizer from tripping over -inf
            # from the attention mask
            softmax_input_quant=None,
            # Quantize the input projections weights as configured
            in_proj_weight_quant=weight_quantizer(bits, _signed=True),
            # Quantize the bias of the input projections as configured
            in_proj_bias_quant=bias_quantizer(bits, _signed=True),
            # No quantization in front of the input projections as this is
            # either done by a standalone quantizer preceding the whole block
            in_proj_input_quant=None,

            # Quantize the output projections weights as configured
            out_proj_weight_quant=weight_quantizer(bits, _signed=True),
            # Quantize the bias of the output projections as configured
            out_proj_bias_quant=bias_quantizer(bits, _signed=True),
            # Quantize the input to the output projection as configured
            out_proj_input_quant=act_quantizer(bits, _signed=True),

            # Quantizer the key after projections as configured
            k_transposed_quant=act_quantizer(bits, _signed=True),
            # Quantize the queries after projections as configured
            q_scaled_quant=act_quantizer(bits, _signed=True),
            # Quantize the values after projection as configured
            v_quant=act_quantizer(bits, _signed=True),

            # No output quantization for now, as stacking multiple layers
            # results in multiple multi-thresholds in succession
            out_proj_output_quant=None,

            # Return the quantization parameters so the next layer can
            # quantize the bias
            return_quant_tensor=True
        )
        # Residual branch addition skipping over the attention layer
        self.residual_sdp = QuantEltwiseAdd(
            # Shared input activation quantizer such that the scales at both
            # input branches are identical. This allows floating point scale
            # factor to be streamlined past the add-node.
            input_quant=act_quantizer(bits, _signed=True),
            # Disable the output quantizer after the add operation. Output of
            # the add will have one more bit than the inputs, which is probably
            # fine and does not require re-quantization.
            output_quant=None,
            # Pass quantization information on to the next layer.
            return_quant_tensor=True
        )
        # Normalization following the attention layer
        self.norm_sdp = torch.nn.Sequential(
            # Select the normalization layer implementation
            get_norm(key=norm, normalized_shape=emb_dim),
            # No quantizer to avoid consecutive quantizer in the MLP residual
            # branch. See input quantizer in front of the first MLP layer.
        )

        # Quantized MLP following the scaled dot-product attention
        self.mlp = torch.nn.Sequential(
            # Quantize the inputs to the MLP block. Placed here to not have this
            # at the input of the residual branch.
            QuantIdentity(
                # Quantize at the output
                act_quant=act_quantizer(bits, _signed=True),
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            ),
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
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # No input quantizer as this is directly preceded by a
                # standalone quantizer
                input_quant=None,
                # Not output quantizer as this is directly followed by a
                # quantized ReLU activation taking care of quantization
                output_quant=None,
                # Return the quantization parameters so the next layer can
                # quantize the bias
                return_quant_tensor=True
            ),
            # Use the ReLU activation function instead of the more commonly used
            # GELU, as the latter is not mapped easily to hardware with FINN
            QuantReLU(
                # Note: ReLU must be quantized to unsigned representation
                act_quant=act_quantizer(bits, _signed=False),
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
                weight_quant=weight_quantizer(bits, _signed=True),
                # Quantize the bias to the same representation as all other
                # layers
                bias_quant=bias_quantizer(bits, _signed=True),
                # No input quantizer as the inputs are already quantized by the
                # preceding ReLU layer
                input_quant=None,
                # Not output quantizer as this is directly followed by a
                # quantized element-wise addition taking care of quantization
                output_quant=None,
                # Pass quantization information on to the next layer.
                return_quant_tensor=True
            ),
        )
        # Residual branch addition skipping over the MLP layer
        self.residual_mlp = QuantEltwiseAdd(
            # Shared input activation quantizer such that the scales at both
            # input branches are identical. This allows floating point scale
            # factor to be streamlined past the add-node.
            input_quant=act_quantizer(bits, _signed=True),
            # Disable the output quantizer after the add operation. Output of
            # the add will have one more bit than the inputs, which is probably
            # fine and does not require re-quantization.
            output_quant=None,
            # Pass quantization information on to the next layer.
            # Note: Not for the last layer to allow this to be combined with
            # standard pytorch calls like .detach() or .numpy(), which are
            # not directly available on QuantTensor.
            return_quant_tensor=True
        )
        # Normalization following the attention layer
        self.norm_mlp = torch.nn.Sequential(
            # Select the normalization layer implementation
            get_norm(key=norm, normalized_shape=emb_dim),
            # No quantizer to avoid consecutive quantizer in the SDP residual
            # branch
        )
        # Generate the attention mask according to configuration
        self.mask = get_mask(mask, seq_len)

    # Forward pass through the transformer block
    def forward(self, x):
        # Move the mask to the same device as the input, just in case...
        mask = self.mask.to(x.device) if self.mask is not None else None
        # Quantize the input to the attention block
        q = self.sdp_input_quant(x)
        # Scaled dot-product attention with residual branch and normalization
        x = self.norm_sdp(
            self.residual_sdp(x, self.sdp(q, q, q, attn_mask=mask)[0])
        )
        # MLP layer with residual branch and normalization
        return self.norm_mlp(self.residual_mlp(x, self.mlp(x)))


# Quantized binary positional encoding layer
class QuantBinaryPositionalEncoding(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(self, input_quant, output_quant, return_quant_tensor):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Adds the quantized input and positional encoding
        self.add = QuantEltwiseAdd(
            # Input quantization to be applied to the input as well as the
            # positional encodings
            input_quant=input_quant,
            # Quantize the outputs after adding input and positional encoding
            output_quant=output_quant,
            # Returns quantization information to the next layer
            return_quant_tensor=return_quant_tensor
        )

    # Forward pass adding positional encoding to the input tensor
    def forward(self, x):
        # Get the size of the inputs to dynamically generate encodings of the
        # same size
        _, seq, emb = x.shape
        # Binary positional encoding fills the embedding dimension with the bit
        # pattern corresponding to the position in the sequence
        pos = torch.as_tensor([
            [(n & (1 << bit)) >> bit for bit in range(emb)] for n in range(seq)
        ])
        # Move the encoding tensor to the same device as the input tensor
        pos = pos.to(x.device, dtype=x.dtype)
        # Add the quantized encoding tp the quantized input
        #   Note: Convert encoding to bipolar representation
        return self.add(x, 2 * pos - 1)


# Unpacks the standard PyTorch tensor from a brevitas QuantTensor
def unpack_from_quant(tensor: torch.Tensor | QuantTensor):
    # If this is a QuantTensor we can extract the wrapped tensor
    if isinstance(tensor, QuantTensor):
        # The underlying tensor is wrapped as the value attribute
        return tensor.value
    # Assume this is already a plain PyTorch tensor
    return tensor


# Dummy transformer encoder model
class DummyTransformer(torch.nn.Module):
    # Initializes the model and registers the module parameters
    def __init__(
            self,
            # Number of layers of attention blocks
            num_layers,
            # Number of attention heads per block
            num_heads,
            # Size of embedding dimension going into/out of the attention block
            emb_dim,
            # Size of MLP dimension in each attention block
            mlp_dim,
            # Length of the input sequence, i.e., context size
            seq_len,
            # Enables bias term added to Linear layers
            bias,
            # Quantization bit-width: For now all layers are quantized to the
            # same bit-width
            bits,
            # Type of normalization layer to use in the transformer blocks
            #   Options are: layer-norm, batch-norm and none
            norm="none",
            # Type of attention mask to use: 'none', 'causal' or 'const'
            mask="none"
    ):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Positional encoding layer at the input
        self.pos = QuantBinaryPositionalEncoding(
            # Quantize the inputs to the positional encoding to the same
            # bit-width as the input
            input_quant=act_quantizer(bits, _signed=True),
            # Quantize the sum of input and positional encoding to the same
            # bit-width as the input
            output_quant=None,
            # Pass quantization information on to the next layer
            return_quant_tensor=True
        )

        # Sequence of num_layers transformer encoder blocks
        self.encoder = torch.nn.Sequential(*[
            TransformerBlock(
                num_heads, emb_dim, mlp_dim, seq_len, bias, norm, mask, bits
            ) for _ in range(num_layers)
        ])

    # Model forward pass taking an input sequence and returning a single set of
    # class probabilities
    def forward(self, x):
        # Add positional encoding to the input and feed through the encoder
        # stack
        # Note: Get the wrapped value out of the QuantTensor to have only a
        # single output from the model.
        return unpack_from_quant(self.encoder(self.pos(x)))
