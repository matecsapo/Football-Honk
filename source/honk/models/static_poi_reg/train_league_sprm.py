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
    model = Static_Poi_Reg_Model(model_name)
    model.Add_Data(results_data.Retrieve(league, season))
    model.Process_Data()
    model.Split_Train_Test(datetime.now())
    model.Train_Model()
    # return model
    return model
# goose train league-sprm [league] [season]
@train_league_sprm.cli
def train_league_sprm_cli(league : str, season : int, save: str = typer.Option(None, "--save", flag_value= ".", help ="Save to specified path")):
    # convert league + season
    league = League(league)
    season = Season(season)
    # train model
    model : Model = train_league_sprm(league, season)
    # if requested, save model
    if save:
        model.save_model_fgm(Path(save))