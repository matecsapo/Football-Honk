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

# scheduled + automated refresh of all Football-Honk modelled
# publicizes updated modelling via git commit + pushing changes
@refresh_operations.operation("automated-git-workflow", "refresh operation invoked by automated git workflow")
def refresh_automated():
    import soccerdata as sd
    from goose.data.pull_data import Standings_Data
    from goose.data.goose_data_structures import League_Table
    import pandas as pd
    def standings_via_US_reconstruction(self, league, season) -> League_Table:
        us = sd.Understat(league, season, proxy=None, no_cache=False, no_store=False)
        data = us.read_team_match_stats()
        home = data[['home_team', 'home_points', 'home_goals', 'away_goals']].rename(
            columns={'home_team': 'Team', 'home_points': 'Pts', 'home_goals': 'GF', 'away_goals': 'GA'}
        )
        away = data[['away_team', 'away_points', 'away_goals', 'home_goals']].rename(
            columns={'away_team': 'Team', 'away_points': 'Pts', 'away_goals': 'GF', 'home_goals': 'GA'}
        )
        table = pd.concat([home, away]).groupby('Team').agg(
            MP=('Pts', 'count'),
            Pts=('Pts', 'sum'),
            GF=('GF', 'sum'),
            GA=('GA', 'sum')
        ).reset_index()
        table['GD'] = table['GF'] - table['GA']
        standings = table[['Team', 'MP', 'Pts', 'GD']].sort_values(
            by=['Pts', 'GD'], ascending=False
        ).reset_index(drop=True)
        standings = League_Table(standings)
        return standings
    Standings_Data.set_source(standings_via_US_reconstruction)
    refresh_all()