# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas quantization layer
from brevitas.nn import QuantIdentity
# Brevitas to QONNX model export
from brevitas.export import export_qonnx


# Scaled Dot-Product Attention operator with quantization layers placed in
# between the matmul and softmax operations
#   Note: This does not contain any of the input or output projections
class ScaledDotProductAttention(torch.nn.Module):
    # Initializes the module parameters and state
    def __init__(self):
        # Initialize the PyTorch Module superclass
        super().__init__()

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
        # Scale derived from embedding dimension size
        scale = torch.sqrt(torch.as_tensor(query.shape[-1]))
        # Multiply queries and keys and quantize the result
        qk = self.qk_quant(torch.bmm(query, key.transpose(-2, -1)) / scale)
        # Softmax-normalization of the attention weights
        a = self.a_quant(torch.softmax(qk, dim=-1))
        # Multiply attention weights and values and quantize the result
        return self.av_quant(torch.bmm(a, value))


# Script entrypoint
if __name__ == '__main__':
    # Seed the pytorch rng to always get the same model and dummy inputs
    torch.manual_seed(1)
    # Create a dummy attention module
    attention = ScaledDotProductAttention()
    # Sample quantized random query, key, value tensors
    q = torch.rand(1, 24, 8)
    k = torch.rand(1, 24, 8)
    v = torch.rand(1, 24, 8)
    # Compute attention output
    o = attention(q, k, v)
    # Export the model graph to QONNX
    export_qonnx(attention, (q, k, v), "attention.onnx")
