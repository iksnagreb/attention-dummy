# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas quantized variants of PyTorch layers
from brevitas.nn import QuantIdentity, QuantMultiheadAttention
# Brevitas to QONNX model export
from brevitas.export import export_qonnx
# Brevitas quantizer
from brevitas.quant import Int8ActPerTensorFloat


# Scaled Dot-Product Attention operator with quantization layers placed in
# between the matmul and softmax operations
#   Note: This does not contain any of the input or output projections
class ScaledDotProductAttention(torch.nn.Module):
    # Initializes the module parameters and state
    def __init__(self):
        # Initialize the PyTorch Module superclass
        super().__init__()

        # Input quantization
        self.q_quant = QuantIdentity()
        self.k_quant = QuantIdentity()
        self.v_quant = QuantIdentity()

        # TODO: Make type and bit-width of quantization configurable

        # Quantizes the output of the query-key matmul
        self.qk_quant = QuantIdentity()
        # Quantizes the output of the softmax normalization
        self.a_quant = QuantIdentity()
        # Quantizes the output of the attention-value matmul
        self.av_quant = QuantIdentity()

    # Attention forward pass: Compute attention weights from queries and keys
    # and applies them to the values
    def forward(self, query, key, value):
        # Quantize the model inputs
        query = self.q_quant(query)
        key = self.k_quant(key)
        value = self.v_quant(value)
        # Scale derived from embedding dimension size
        scale = torch.sqrt(torch.as_tensor(query.shape[-1]))
        # Multiply queries and keys and quantize the result
        qk = self.qk_quant(torch.bmm(query, key.transpose(-2, -1)) / scale)
        # Softmax-normalization of the attention weights
        a = self.a_quant(torch.softmax(qk, dim=-1))
        # Multiply attention weights and values and quantize the result
        return self.av_quant(torch.bmm(a, value))


# Minimal example of a model which could be considered as a transformer
class DummyTransformer(torch.nn.Module):
    # Initializes and registers the module parameters and state
    def __init__(self, embed_dim, num_heads, num_layers=2, batch_first=True):
        # Initialize the PyTorch Module superclass
        super().__init__()
        # Stack multiple scaled dot product attention operators
        self.attention_blocks = torch.nn.ModuleList(num_layers * [
            # Each block is an instance of quantized scaled dot-product
            # attention from brevitas
            QuantMultiheadAttention(
                # Size of the embedding dimension (input and output)
                embed_dim=embed_dim,
                # Number of attention heads
                num_heads=num_heads,
                # Enable a bias added to the input and output projections
                bias=True,
                # Layout of the inputs:
                #   Batch x Sequence x Embedding (batch-first, True)
                #   Sequence x Batch x Embedding (batch-second, False)
                batch_first=batch_first,
                # If query, key and value input are the same, packed input
                # projections use a single, large linear projection to produce
                # the actual query, key and value inputs. Otherwise, use
                # separate linear projections on each individual input.
                packed_in_proj=True,
                # Brevitas has this as an unsigned quantizer by default, but
                # finn can only handle signed quantizer
                attn_output_weights_quant=Int8ActPerTensorFloat,
                # Insert an additional quantizer in front ot the softmax. In our
                # finn custom-op, this will be matched to the quantizer
                # following the query and key matmul.
                softmax_input_quant=Int8ActPerTensorFloat
            ),
        ])

    # Model forward pass doing self attention, i.e, distributing a single input
    # to the query, key and value inputs of the attention operator
    def forward(self, x):  # noqa: Shadows name 'x' from outer scope
        # There are multiple blocks of attention
        for block in self.attention_blocks:
            # Distribute input to all three attention inputs and use output as
            # next blocks input
            x, _ = block(x, x, x)  # noqa: Shadows name 'x' from outer scope
        # Return the output of the final block as the global output
        return x


# Script entrypoint
if __name__ == '__main__':
    # Seed the pytorch rng to always get the same model and dummy inputs
    torch.manual_seed(1)
    # Create a dummy attention module
    attention = DummyTransformer(
        embed_dim=8, num_heads=4, num_layers=2, batch_first=True
    )
    # Sample random input tensor in batch-first layout
    x = torch.rand(1, 64, 8)
    # Compute attention output
    o = attention(x)
    # Export the model graph to QONNX
    export_qonnx(attention, (x, ), "attention.onnx")
