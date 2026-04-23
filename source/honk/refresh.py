# for defining an operations folder for storing refresh operations
from goose.operation.built_in_operations.goose_operations import goose_operations
from goose.name_standardization import standardize_league_name

# for training models and building projections
from honk.models.model_train_scripts.train_sprm import train_sprm
from honk.projections.projection_build_scripts.projection import project

# for building projections according to Football-Honk's config
from honk.config import modelled_leagues, flagship_models

# operations folder for storing refresh operations
# goose refresh
refresh_operations = goose_operations.create_subfolder("refresh", "refresh modelling")

# operation for refreshing specified league:
    # refresh(trains) its flagship model
    # refresh(builds) its projections
# goose refresh league [league]
@refresh_operations.operation("league", "refresh modelling for specified league")
def refresh_league(league : str):
    league = standardize_league_name(league)
    print(f"Refreshing {league} modelling and projections...")
    # determine league's corresponding flagship Football-Honk model
    model_name, train_function = flagship_models[league]
    # refresh (train) sprm model
    train_function(league)
    # refresh (build) projections
    project(league, model_name)

# operation for refreshing all leagues modelled by Football-Honk:
# goose refresh all
@refresh_operations.operation("all", "refresh modelling for all supported leagues")
def refresh_all():
    print(f"Refreshing all modelling and projections...")
    for league in modelled_leagues:
        refresh_league(league)