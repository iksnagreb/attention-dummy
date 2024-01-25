# Experiment metrics are saved as YAML files
import yaml
# Pandas to handle the reports as table, i.e., DataFrame
import pandas as pd

# Script entrypoint
if __name__ == "__main__":
    # Open the report file
    with open("build/report/estimate_layer_resources_hls.json") as file:
        # Load the JSON formatted report
        report = pd.read_json(file, orient="index")
    # Generate a summary of the total resources
    summary = report.sum()
    # Dump the metrics dictionary as yaml
    with open("metrics.yaml", "w") as file:
        # Convert the dataframe to a dictionary which can be dumped into YAML
        yaml.safe_dump(summary.to_dict(), file)
