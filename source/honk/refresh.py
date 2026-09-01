# for defining an operations folder for storing refresh operations
from goose.operation.built_in_operations.goose_operations import goose_operations
from goose.data.goose_data_structures.identifiers import League, Season

# for training models and building projections
from honk.projection_build_scripts.projection import project
from goose.model import Model
from goose.operation.built_in_operations.utilities import load_model
from honk.models.static_poi_reg.train_league_sprm import train_league_sprm
from honk.models.weighted_poi_reg.weighted_poi_reg_train_scripts import wprm_update

# for building projections according to Football-Honk's config
import honk.config as config
from pathlib import Path
import typer

# operations folder for storing refresh operations
# goose refresh
refresh_operations = goose_operations.create_subfolder("refresh", "refresh modelling")

# operation for refreshing specified model:
    # refresh(updates/trains) model
    # refresh(builds) its projections
# goose refresh model [league]
@refresh_operations.operation("model", automatically_supports_cli = True, description = "refresh modelling for specified model")
def refresh_model(model_name : str):
    # determine model's refresh function
    league, active_inactive, refresh_function = config.models[model_name]
    # refresh model
    refresh_function(model_name)
    # load refreshed model
    model, model_name = load_model(model_name)
    # build projections
    project(league, config.current_season, model, Path.cwd() / "honk/live/projections")

# operation for refreshing a given production model
# goose train sprm-refresh-live [model_name]
@refresh_operations.operation("refresh-live-sprm", automatically_supports_cli = False, description = "Refresh live sprm model given its name")
def refresh_live_sprm(model_name : str):
    # retrieve model's league
    league, _, _ = config.models[model_name]
    # train model
    model : Model = train_league_sprm(league, config.current_season)
    # save sprm model to honk/live/models
    model.save_model_fgm(Path.cwd() / "honk/live/models")

# operation for refreshing a given live model
# goose train wprm-refresh-live [model_name]
@refresh_operations.operation("refresh-live-wprm", automatically_supports_cli = False, description = "Refresh live wprm model given its name")
def refresh_live_wprm(model_name : str):
    # retrieve model's league
    league, _, _ = config.models[model_name]
    # load desired wprm model
    model, model_name = load_model(model_name)
    # train model
    model : Model = wprm_update(league, config.current_season, model)
    # save updated model to honk/live/models
    model.save_model_fgm(Path.cwd() / "honk/live/models")

# operation for refreshing all leagues modelled by Football-Honk:
# goose refresh all
@refresh_operations.operation("all", automatically_supports_cli = True, description = "refresh modelling for all supported leagues")
def refresh_all():
    print(f"Refreshing all Active Models and Projections...")
    for model_name, (_, active_inactive, _) in config.models.items():
        if active_inactive == True:
            refresh_model(model_name)
            typer.echo("")

# operation for running scheduled + automated refresh of all Football-Honk modeling
# uses standings-via-understats-reconstruction as opposed to default of ESPN
    # since ESPN can sometimes block automated scrapers
@refresh_operations.operation("automated-refresh", automatically_supports_cli = True, description = "refresh operation invoked by automated git workflow")
def refresh_automated():
    # set alternative standings retrieval source
    from goose.data.built_in_data_types.standings_data import standings_data
    standings_data.Set_Source("Understats(Reconstruction)")
    typer.echo("Standings data source set to : Understats(Reconstruction)")
    # run full refresh of all modelling
    refresh_all()