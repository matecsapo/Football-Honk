# for defining an operation for building projections
from goose.operation.built_in_operations.goose_operations import goose_operations
from goose.operation.built_in_operations.utilities import load_model
from goose.operation.built_in_operations.forecast_operations import expectation, monte_carlo
from goose.operation.built_in_operations.prediction_operations import predict_remaining
from goose.name_standardization import standardize_league_name
from pathlib import Path
import json
from datetime import datetime

# operation for building projection for specified league using specified model
# goose project [league] [model_name]
@goose_operations.operation("project", "builds a projection")
def project(league : str, model_name : str):
    # standardize league name
    league = standardize_league_name(league)
    # folder for storing produced projections in honk/projections/[league]
    folder = Path(__file__).parent.parent / league
    # build forecasts for (league, model)
    expectation(league, model_name, save = folder)
    print()
    monte_carlo(league, model_name, save = folder)
    # predict out all remaining games
    print()
    predict_remaining(league, model_name, save = folder)
    # save projections identification information
    projection_identification = {
        "Generating Model" : model_name,
        "Timestamp" : datetime.now().isoformat()
    }
    with open(folder / f"{league}_projection_identification.json", "w") as f:
        json.dump(projection_identification, f)
