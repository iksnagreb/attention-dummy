# Use the DVC api for loading the YAML parameters
import dvc.api
# Progressbar
from tqdm import trange
# For saving numpy array data
import numpy as np
# PyTorch base package: Math and Tensor Stuff
import torch
# Brevitas to QONNX model export
from brevitas.export import export_qonnx

# Dummy model for testing
from model import DummyTransformer
# Seeding RNGs for reproducibility
from utils import seed


# Check whether a layer is a normalization layer of some supported type
def is_norm_layer(module):
    # Set of normalization layer (bases) which maybe need to be patched
    norm_layers = {
        # All BatchNorm and InstanceNorm variants derive from this baseclass
        torch.nn.modules.batchnorm._NormBase,  # noqa: Access to _NormBase
        # LayerNorm has a unique implementation
        torch.nn.LayerNorm,
    }
    # Check the module against all supported norm layer types
    return any(isinstance(module, norm) for norm in norm_layers)


# Fixes export issues of normalization layers with disabled affine parameters.
# Somehow the export to ONNX trips when it encounters the weight and bias tensor
# to be 'None'.
def patch_non_affine_norms(model: torch.nn.Module):  # noqa: Shadows model
    # Iterate all modules in the model container
    for name, module in model.named_modules():
        # If the module is a normalization layer it might require patching the
        # affine parameters
        if is_norm_layer(module):
            # Check whether affine scale parameters are missing
            if hasattr(module, "weight") and module.weight is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_var"):
                    # Patch the affine bias by all 1 tensor of the same shape,
                    # type and device as the running variance
                    module.weight = torch.nn.Parameter(
                        torch.ones_like(module.running_var)
                    )
            # Check whether affine bias parameters are missing
            if hasattr(module, "bias") and module.bias is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_mean"):
                    # Patch the affine bias by all 0 tensor of the same shape,
                    # type and device as the running mean
                    module.bias = torch.nn.Parameter(
                        torch.zeros_like(module.running_var)
                    )
    # Return the patched model container
    return model


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml")
    # Seed all RNGs
    seed(params["seed"])
    # Make PyTorch behave deterministically if possible
    torch.use_deterministic_algorithms(mode=True, warn_only=True)
    # Create a model instance from the configuration parameters
    model = DummyTransformer(**params["model"])
    # Get the configured sequence length and embedding dimension to generate
    # test inputs
    seq, dim = params["model"]["seq_len"], params["model"]["emb_dim"]
    # No gradient accumulation for calibration passes required
    with torch.no_grad():
        # Check whether GPU training is available and select the appropriate
        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move the model to the training device
        model = model.to(device)
        # Multiple passes of calibration might be necessary for larger/deep
        # models
        for _ in trange(0, params["calibration_passes"], desc="calibrating"):
            # Pass random data through the model to "calibrate" dummy quantizer.
            # Large batch to have more calibration samples. Otherwise, there is
            # too much deviation between this calibration and the verification
            # samples.
            model(torch.rand(128, seq, dim, device=device))
        # Move the model back to the CPU
        model = model.cpu()
    # Prevent export issue for missing affine normalization parameters
    model = patch_non_affine_norms(model)
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
    export_qonnx(model, (x,), "attention.onnx", **params["export"])
