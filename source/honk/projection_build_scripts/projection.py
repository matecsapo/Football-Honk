# for defining an operation for building projections
from goose.operation.built_in_operations.goose_operations import goose_operations
from goose.operation.built_in_operations.utilities import load_model
from goose.model import Model
from goose.forecast.forecast import Forecast
from goose.operation.built_in_operations.forecast_operations import expectation, monte_carlo
from goose.operation.built_in_operations.prediction_operations import predict_remaining
from goose.data.goose_data_structures.identifiers import League, Season
from goose.data.goose_data_structures.game_storage import Games
from pathlib import Path
import json
from datetime import datetime
import typer

# operation for building projection for specified league using specified model
@goose_operations.operation("project", description = "builds a projection")
def project(league : League, season : Season, model : Model, save : Path):
    typer.echo(f"Producing Projections for {league.league} via {model.model_name}...")
    # folder for storing produced projections in honk/live/projections/[league]
    folder = save / f"{model.model_name}_projection"
    folder.mkdir(parents=True, exist_ok=True)
    # build forecasts for (league, model)
    exp : Forecast = expectation(league, season, model)
    exp.Save_Forecast(folder)
    mc : Forecast = monte_carlo(league, season, model)
    mc.Save_Forecast(folder)
    # predict out all remaining games
    rem_pred : Games = predict_remaining(league, season, model)
    rem_pred.save_data(folder / "remaining_games_predictions.csv")
    # save projections identification information
    projection_identification = {
        "Generating Model" : model.model_name,
        "Timestamp" : datetime.now().isoformat()
    }
    with open(folder / f"{league.league}_projection_identification.json", "w") as f:
        json.dump(projection_identification, f)
# goose project [league] [season] [model_name] [save_path]
@project.cli
def project_cli(league : str, season : int, model_name : str, save: str = typer.Option(".", "--save", "-s", help="Path to save model")):
    # convert league + season
    league = League(league)
    season = Season(season)
    # load desired model
    model, model_name = load_model(model_name)
    # produce projections
    project(league, season, model, Path(save))
    typer.echo(f"Saved to {save}")
