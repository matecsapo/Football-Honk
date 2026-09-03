# for defining a model training operation
from honk.models.train_scripts import model_train_operations

# for loading models
from goose.operation.built_in_operations.utilities import load_model

# for training weighted poisson regression models
from goose.data.goose_data_structures.identifiers import League, Season
from goose.data.goose_data_structures.game_storage import Games
from goose.model import Model
from goose.data.goose_data_structures.standings_storage import Table
from honk.models.weighted_poi_reg.weighted_poi_reg_model import Weighted_Poi_Reg_Model
from goose.data.built_in_data_types.results_data import results_data
from goose.data.built_in_data_types.standings_data import standings_data
from datetime import datetime
from pathlib import Path
import typer

# operation for base fitting weighted_poi_reg model on games of previous season
    # only games involving teams present in current season are fitted on
@model_train_operations.operation("wprm-basefit", description = "base fit weighted poisson regression model for specified league on previous season data")
def wprm_basefit(league : League, season : Season):
    # identify given season teams
    season_results : Games = results_data.Retrieve(league, season)
    season_teams = season_results.Teams_Involved()
    # retrieve previous season's results data
    previous_season_results : Games = results_data.Retrieve(league, Season(season.start_year - 1))
    previous_season_teams = previous_season_results.Teams_Involved()
    # filter for only games involving current_season_teams
    condition = lambda g: (g.home_team in season_teams) and (g.away_team in season_teams)
    previous_season_results = previous_season_results.Filter(condition)
    # base fit model on previous season_results
    model_name = league.league + "_wprm"
    print(f"Basefitting {league.league} WPRM model...")
    model = Weighted_Poi_Reg_Model(model_name, league, season)
    model.fit(previous_season_results)
    # Impute newly promoted (i.e. that weren't present last season)
    new_teams = season_teams - previous_season_teams
    for team in new_teams:
        model.impute_team_as_median(team)
    # return model
    return model
# goose train wprm-basefit [league] [season] Flag[--save]
@wprm_basefit.cli
def wprm_basefit_cli(league : str, season : int, save: str = typer.Option(".", "--save", "-s", help="Path to save model")):
    # convert league + season
    league = League(league)
    season = Season(season)
    # train model
    model : Model = wprm_basefit(league, season)
    # save model
    model.save_model_fgm(Path(save))
    typer.echo(f"Saved to {save}")

# operation for updating existing league-specific weighted_poi_reg model given additional games
@model_train_operations.operation("wprm-update", description = "update existing league-specific wprm model with freshest games")
def wprm_update(league : League, season : Season, model : Model):
    # Grab most recent match-time of data present in model
    existing_games = model.fit_games
    most_recent_match_time = existing_games.to_dataframe()["date"].max()
    # Grab fresh games
    new_games : Games = results_data.Retrieve(league, season)
    new_games = new_games.Filter(lambda x : x.date > most_recent_match_time)
    # Update model
    print(f"Updating {model.model_name} WPRM model...")
    model.update(new_games)
    # return model
    return model
# goose train wprm-update [league] [season] Flag[--save]
@wprm_update.cli
def wprm_update_cli(league : str, season : int, model_name : str, save : str = typer.Option(".", "--save", "-s", help="Path to save model")):
     # convert league + season
    league = League(league)
    season = Season(season)
    # load desired model
    model, model_name = load_model(model_name)
    # update model
    model : Model = wprm_update(league, season, model)
    # save model
    model.save_model_fgm(Path(save))
    typer.echo(f"Saved to {save}")

# operation for refreshing a given live sprm production model from honk/live/models
import honk.refresh as refresh
import honk.config as config
# goose refresh refresh-live-sprm [model]
@refresh.refresh_operations.operation("refresh-live-wprm", automatically_supports_cli = True, description = "Refresh live wprm model given its name")
def refresh_live_wprm(model_name : str):
    # load desired wprm model
    model, model_name = load_model(model_name)
    # league
    league = model.league
    # update model
    model : Weighted_Poi_Reg_Model = wprm_update(league, config.current_season, model)
    # save updated model to honk/live/models
    model.save_model_fgm(Path.cwd() / "honk/live/models")

# operation for refreshing a given live wprm model's projections in honk/live/projections
from honk.projection_build_scripts.projection import project
# goose refresh refresh-live-wprm-projections [model]
@refresh.refresh_operations.operation("refresh-live-wprm-projections", automatically_supports_cli = True, description = "Refresh live wprm model's projections")
def refresh_live_wprm_projections(model_name : str):
    # load desired wprm model
    model, model_name = load_model(model_name)
    # model's target league
    league = model.league
    # produce projections
    project(league, config.current_season, model, Path.cwd() / "honk/live/projections")