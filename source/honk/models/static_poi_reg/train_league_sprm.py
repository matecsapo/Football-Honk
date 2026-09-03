# for defining a model training operation
from honk.models.train_scripts import model_train_operations

# for training sprm models
from goose.data.goose_data_structures.identifiers import League, Season
from goose.model import Model
from honk.models.static_poi_reg.static_reg_poi_model import Static_Poi_Reg_Model
from goose.data.built_in_data_types.results_data import results_data
from goose.operation.built_in_operations.utilities import load_model
from datetime import datetime
from pathlib import Path

# for typer support
import typer

# operation for training sprm model for specified league-style competition
@model_train_operations.operation("league-sprm", description = "train sprm model for specified league")
def train_league_sprm(league : League, season : Season):
    # build model for specified league based on most recent data
    model_name = league.league + "_sprm"
    print(f"Training {league.league} SPRM model...")
    model = Static_Poi_Reg_Model(model_name, league, season)
    model.Add_Data(results_data.Retrieve(league, season))
    model.Process_Data()
    model.Split_Train_Test(datetime.now())
    model.Train_Model()
    # return model
    return model
# goose train league-sprm [league] [season]
@train_league_sprm.cli
def train_league_sprm_cli(league : str, season : int, save: str = typer.Option(".", "--save", "-s", help="Path to save model")):
    # convert league + season
    league = League(league)
    season = Season(season)
    # train model
    model : Model = train_league_sprm(league, season)
    # if requested, save model
    if save:
        model.save_model_fgm(Path(save))

# operation for refreshing a given live sprm production model from honk/live/models
import honk.refresh as refresh
import honk.config as config
# goose refresh refresh-live-sprm [model]
@refresh.refresh_operations.operation("refresh-live-sprm", automatically_supports_cli = True, description = "Refresh live sprm model given its name")
def refresh_live_sprm(model_name : str):
    # load desired wprm model
    model, model_name = load_model(model_name)
    # league
    league = model.league
    # train model
    model : Static_Poi_Reg_Model = train_league_sprm(league, config.current_season)
    # save sprm model to honk/live/models
    model.save_model_fgm(Path.cwd() / "honk/live/models")

# operation for refreshing a given live sprm model's projections in honk/live/projections
from honk.projection_build_scripts.projection import project
# goose refresh refresh-live-sprm-projections [model]
@refresh.refresh_operations.operation("refresh-live-sprm-projections", automatically_supports_cli = True, description = "Refresh live sprm model's projections")
def refresh_live_sprm_projections(model_name : str):
    # load desired wprm model
    model, model_name = load_model(model_name)
    # model's target league
    league = model.league
    # produce projections
    project(league, config.current_season, model, Path.cwd() / "honk/live/projections")