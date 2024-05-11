# Experiment metrics are saved as YAML files
import yaml
# Use the DVC api for loading the YAML parameters
import dvc.api
# Find all files matching a pattern in directory
import glob


# Script entrypoint
if __name__ == "__main__":
    # Load the parameters file
    params = dvc.api.params_show("params.yaml")
    # Collect all verification output filenames
    outputs = glob.glob(
        f"{params['build']['output_dir']}/verification_output/*.npy"
    )
    # Extract the verification status for each verification output by matching
    # to the SUCCESS string contained in the filename
    status = all([
        out.split("_")[-1].split(".")[0] == "SUCCESS" for out in outputs
    ])
    # Dump the verification status as yaml
    with open("verification.yaml", "w") as file:
        # Construct a dictionary reporting the verification status as string
        yaml.safe_dump(
            {"verification": {True: "success", False: "fail"}[status]}, file
        )
