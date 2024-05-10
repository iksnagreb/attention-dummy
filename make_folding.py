# Use the DVC api for loading the YAML parameters
import dvc.api
# JSON for folding configuration
import json

# Seeding RNGs for reproducibility
from utils import seed


# Generates a folding configuration for a transformer model of multiple layers
# each with multiple heads
def make_folding(**_):
    # Start filling the folding configuration with defaults applied to all
    # operators
    folding = {  # noqa: Shadows outer scope
        "Defaults": {
            # Let the tools automatically decide how to implement memory for
            # FIFO buffers and MVAU weights
            "ram_style": ["auto", [
                "StreamingFIFO_hls", "StreamingFIFO_rtl", "MVAU_hls", "MVAU_rtl"
            ]]
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
