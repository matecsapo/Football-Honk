# for defining a model training operation
from honk.models.train_scripts import model_train_operations

# for training weighted poisson regression models
from goose.data.goose_data_structures.identifiers import League, Season
from goose.data.goose_data_structures.game_storage import Games
from goose.data.goose_data_structures.standings_storage import Table
from honk.models.weighted_poi_reg.weighted_poi_reg_model import Weighted_Poi_Reg_Model
from goose.data.built_in_data_types.results_data import results_data
from goose.data.built_in_data_types.standings_data import standings_data
from datetime import datetime
from pathlib import Path
from honk.config import current_season

# operation for base fitting weighted_poi_reg model on last 10 games of previous season
    # The last 10 games of the previous season are used
    # only games involving teams present in current season are fitted on
# goose train league-wprm [league]
@model_train_operations.operation("league-basefit-wprm", "base fit weighted poisson regression model for specified league on previous season data")
def train_league_sprm(league : str):
    # standardize league name
    if isinstance(league, str):
        league = League(league)
    # identify current season teams
    current_season_standings : Table = standings_data.Retrieve(league, current_season)
    current_season_teams = current_season_standings.standings["Team"].unique()
    # retrieve last season's results data
    previous_season_results : Games = results_data.Retrieve(league, Season(current_season.start_year - 1))
    # filter for only games involving current_season_teams
    condition = lambda g: (g.home_team in current_season_teams) and (g.away_team in current_season_teams)
    previous_season_results = previous_season_results.Filter(condition)
    # base fit model on previous season_results
    model_name = league.league + "_wprm"
    print(f"Training {model_name}...")
    model = Weighted_Poi_Reg_Model(model_name)
    model.fit(previous_season_results)
    # save model to current directory
    model.save_model_fgm("")
    print(f"Trained and saved {model_name}")