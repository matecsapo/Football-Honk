# for implementing a model
from goose.model import Model

# For data manipulation
from goose.data.goose_data_structures.identifiers import Team, League, Season
from goose.data.goose_data_structures.game_storage import Game, Games, Game_Prediction
import numpy as np
import pandas as pd
import json as json
import os as os
from pathlib import Path
from typing import Self

# For fitting poisson regression models
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, skellam

# initialize
# fit
# update




# State-supporting poisson regression model
# Builds a poisson regression model on XG as an explanation of 41 parameters:
    # att, def values for each 20 teams
    # league-consistent estiamte of home advantage factor
    # model's state can be updated with additional games
        # model updates by completely refitting on all (incl. new) games

@Model.define_model("State Poisson Regression Model")
class State_Poi_Reg_Model(Model):
    # initialize with hyperparameter + model info
    def __init__(self, model_name : str = None, league : League = None, season : Season = None):
        self.model_name = model_name
        self.league = League
        self.season = Season
        # training games to fit the model on
        self.fit_games : Games = None
        # poisson regression model
        self.model = None
        self.model_parameters = None

    # fit model on specified set of games
    def fit(self, fit_games : Games = None):
        # store fit_games, if any passed
        if fit_games != None:
            self.fit_games = fit_games
        
        

    # update/refresh model subject to new additional fit_games
    def update(self, additional_fit_games : Games):
        # store the additional fit_games
        self.fit_games.Add_Games(additional_fit_games)
        # refit model
        self.fit()
    
        
