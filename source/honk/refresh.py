# for defining an operations folder for storing refresh operations
from goose.operation.built_in_operations.goose_operations import goose_operations
from goose.data.goose_data_structures.identifiers import League, Season

# for training models and building projections
from honk.projection_build_scripts.projection import project
from goose.model import Model
from goose.operation.built_in_operations.utilities import load_model

# for building projections according to Football-Honk's config
import honk.config as config
from pathlib import Path
import typer

# operations folder for storing refresh operations
# goose refresh
refresh_operations = goose_operations.create_subfolder("refresh", "refresh modelling")

# operation for refreshing specified live model from honk/live/models:
    # refresh(updates/trains) model
    # refresh(builds) its projections
# goose refresh model [league]
@refresh_operations.operation("model", automatically_supports_cli = True, description = "refresh modelling for specified model")
def refresh_model(model_name : str):
    # determine model's refresh function
    _, refresh_model_op, refresh_projections_op = config.models[model_name]
    # refresh model
    refresh_model_op(model_name)
    # build projections
    refresh_projections_op(model_name)

# operation for refreshing all leagues modelled by Football-Honk:
# goose refresh all
@refresh_operations.operation("all", automatically_supports_cli = True, description = "refresh modelling for all supported leagues")
def refresh_all():
    print(f"Refreshing all Active Models and Projections...")
    for model_name, (active_inactive, _, _) in config.models.items():
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