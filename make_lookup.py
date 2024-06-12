# Use the DVC api for loading the YAML parameters
import dvc.api
# For saving numpy array data
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas to QONNX model export
from brevitas.export import export_qonnx
# Brevitas quantized embedding lookup layer
from brevitas.nn import QuantEmbedding

# Weight quantizer with configurable bit-width and signedness
from model import weight_quantizer
# Seeding RNGs for reproducibility
from utils import seed


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml")
    # Seed all RNGs
    seed(params["seed"])
    # Make PyTorch behave deterministically if possible
    torch.use_deterministic_algorithms(mode=True)
    # Get the configured sequence length and embedding dimension to generate
    # test inputs
    seq, dim = params["model"]["seq_len"], params["model"]["emb_dim"]

    # Create only a quantized lookup layer as the model
    model = QuantEmbedding(
        # Number of different embedding vectors in the lookup table
        num_embeddings=2048,
        # Size of each of the embedding vectors
        embedding_dim=dim,
        # Quantize the elements of the embedding vectors
        weight_quant=weight_quantizer(params["model"]["bits"])
    )

    # No gradient accumulation for calibration passes required
    with torch.no_grad():
        # Multiple passes of calibration might be necessary for larger/deep
        # models
        for _ in range(params["calibration_passes"]):
            # Pass random data through the model to "calibrate" dummy quantizer.
            # Large batch to have more calibration samples. Otherwise, there is
            # too much deviation between this calibration and the verification
            # samples.
            model(torch.randint(low=0, high=64, size=(16384, seq)))

    # Switch model to evaluation mode to have it fixed for export
    model = model.eval()
    # Sample random input tensor in batch-first layout
    x = torch.randint(low=0, high=64, size=(1, seq))
    # Compute attention output
    o = model(x)
    # Save the input and output data for verification purposes later
    np.save("inp.npy", x.detach().numpy())
    np.save("out.npy", o.detach().numpy())
    # Export the model graph to QONNX
    export_qonnx(model, (x,), "lookup.onnx", **params["export"])
