from pathlib import Path
import sys
import json

project_root = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(project_root))

from src.artificial import data_preparation as dp

parameters_file = project_root / "config" / "artificial" / "data_parameters.json"


if __name__ == "__main__":

    with open(parameters_file, "r", encoding="utf-8") as file:
        parameters = json.load(file)

    dataset_directory = dp.prepare_data(parameters, raw=True)

    print(f"Dataset successfully generated: {dataset_directory.name}")