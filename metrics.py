# Experiment metrics are saved as YAML files
import yaml
# Pandas to handle the reports as table, i.e., DataFrame
import pandas as pd

# Script entrypoint
if __name__ == "__main__":
    # Open the configuration file
    with open("params.yaml") as file:
        # Load the configuration from yaml format
        params = yaml.safe_load(file)["metrics"]
    # Open the report file
    with open(params["report"]) as file:
        # Load the JSON formatted report
        report = pd.read_json(file, orient="index")
    # Filter the reported rows according to some regex filter rule
    report = report.filter(regex=params["filter"], axis="rows")
    # Generate a summary of the total resources
    summary = report.sum()
    # Dump the metrics dictionary as yaml
    with open("metrics.yaml", "w") as file:
        # Convert the dataframe to a dictionary which can be dumped into YAML
        yaml.safe_dump(summary.to_dict(), file)
