# for implementing a model
from goose.model import Model

# For data manipulation
from goose.data.goose_data_structures.identifiers import Team, League, Season
from goose.data.goose_data_structures.game_storage import Game, Completed_Game, Game_Simulation, Games, Game_Prediction
import numpy as np
import pandas as pd
import json as json
import os as os
from pathlib import Path
from typing import Self
from datetime import datetime

# For fitting poisson regression models
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, skellam

# Weighted, stateful poisson regression model
# Fits a log-link poisson regression model on xG as an explanation of 41 parameters:
    # att, def values for each team in fit-data set of games
    # league-wide home-advantage-factor parameter
# fit_games datapoints are exponential-decay weighted by halflife
# .update() updates model accounting for additional (i.e. new) games
    # model updates by simply refittin on entirity (i.e. incl. new) of games
@Model.define_model("Stateful Poisson Regression Model")
class Weighted_Poi_Reg_Model(Model):
    # initialize with hyperparameters + model info
    def __init__(self, model_name : str, league : League, season : Season, weight_decay_halflife_days : int = 180):
        super().__init__(model_name)
        # league + season model targets
        self.league = league
        self.season = season
        # training games to fit the model on
        self.fit_games : Games = None
        # hyperparameters
        self.weight_decay_halflife_days = weight_decay_halflife_days
        # poisson regression model
        self.model = None
        self.model_parameters = None

    # fit model on specified set of games
    def fit(self, fit_games : Games[Completed_Game] = None):
        self.fit_games = fit_games
        self.fit_poi_reg_model()

    # update/refresh model subject to new additional fit_games
    def update(self, additional_fit_games : Games):
        # store the additional fit_games
        self.fit_games.Add_Games(additional_fit_games.games)
        # refit model
        self.fit_poi_reg_model()    

    # fit poisson regression model on supplied fit_games
    def fit_poi_reg_model(self):
        # prepare data
        data = self.prepare_data()
        # calculate game_datapoint_weights
        weights = self.game_datapoint_weights(data)
        # estimate model parameters
        self.model = smf.glm(  
            formula="xg ~ team + opponent + h_a", # formula automatically expands categorial home/team columns into one-for-each team
            data = data,
            family = sm.families.Poisson(), # poisson regression
            var_weights = weights
        ).fit()
        # Extract parameters from model
        params = self.model.params
        # Exponentiate out of log-space
        params = np.exp(params)
        # store parameters into clean struct self.Model_Parameters
        self.model_parameters = {}
        # 1. intercept
        self.model_parameters["Intercept"] = params["Intercept"]
        # 2. (all) Team att and def values
        for team in sorted(self.fit_games.Teams_Involved()):
            self.model_parameters[team] = {"att" : params.get(f"team[T.{team}]", 1), # naming structure employed by smf
                                                "def" : params.get(f"opponent[T.{team}]", 1)} # " "
        # 3. h/a_factor
        self.model_parameters["h_a_factor"] = params["h_a"]

    # prepare data from Games container into clean input feature -> target table
    def prepare_data(self):
        # convert from Games container to dataframe
        data = self.fit_games.to_dataframe()
        # keep + clean only neeeded cols
        data = data[["date", "home_team", "away_team", "home_xg", "away_xg"]].reset_index(drop=True)
        data["date"] = pd.to_datetime(data["date"])
        # Duplicate each game (i.e. = datapoint) from each team's (H & A) perspective
        home_perspective = pd.DataFrame(data[["date", "home_team", "away_team", "home_xg"]])
        home_perspective["h_a"] = 1 # true
        home_perspective = home_perspective.rename(columns = {"home_team" : "team", "away_team" : "opponent", "home_xg" : "xg"})
        away_perspective = pd.DataFrame(data[["date", "away_team", "home_team", "away_xg"]])
        away_perspective["h_a"] = 0 # false
        away_perspective = away_perspective.rename(columns = {"away_team" : "team", "home_team" : "opponent", "away_xg" : "xg"})
        # produced table of xg | team, opponent, h/a    
        data = pd.concat([home_perspective, away_perspective], ignore_index=True)
        data = data[["date", "team", "opponent", "h_a", "xg"]]
        return data

    # Calculate weights associated with each fit_games:
    def game_datapoint_weights(self, data : pd.DataFrame):
        # Current datetime
        current_datetime = datetime.today()
        # Calculate weight associated with each game
        days_ago = (current_datetime - data["date"]).dt.days.astype(int)
        weights = 0.5 ** (days_ago / self.weight_decay_halflife_days)
        return weights

    # Imputes (new) team's (att, def) parameters as median of existing teams
    def impute_team_as_median(self, team : Team):
        existing_teams  = self.fit_games.Teams_Involved()
        # Extract all attack and defense parameters for existing teams
        att_values = [self.model_parameters[t]["att"] for t in existing_teams if t in self.model_parameters]
        def_values = [self.model_parameters[t]["def"] for t in existing_teams if t in self.model_parameters]
        # Calculate medians
        median_att = float(np.median(att_values))
        median_def = float(np.median(def_values))
        # Assign to specified team
        self.model_parameters[team] = {
            "att" : median_att,
            "def" : median_def
        }

    # Saves model to a folder self.Model_Name/; includes:
        # Dump file produced directly by smf
        # Json object of Model_Parameters dictionary
        # Model Evaluation statistics
    def save_model(self, model_save_root : str | Path):
        # folder for model storage
        folder = Path(model_save_root)
        os.makedirs(folder, exist_ok=True)
        # smf dump file
        self.model.save(folder / "smfmodel.pkl")
        # json Model_Paramaters dump
        with open(folder / "parameters.json", "w") as f:
            data = {str(k): v for k, v in self.model_parameters.items()}
            json.dump(data, f, indent=4)

    # Loads model saved to folder self.Model_Name/ given path
    @classmethod
    def load_model(cls, model_save_path : str | Path) -> Self:
        model_name = Path(model_save_path).name
        model = cls(model_name)
        # load smf dump
        model.model = sm.load(model_save_path / "smfmodel.pkl")
        # load parameters
        with open(model_save_path / "parameters.json", "r") as f:
            model.model_parameters = {Team(k) if k not in ["Intercept", "h_a_factor"] else k: v for k, v in json.load(f).items()}
        # return loaded model
        return model

    # Produces a Game_Prediction using model for specified game
        # home/away_pred_goals are Poisson distribution expected value = xG
        # home win, away win, draw probabilities derived via skellam distribution comparing home and away xG distributions
    def Predict_Game(self, game : Game):
        # Extract all necessary model ratings
        intercept = self.model_parameters["Intercept"]
        home = self.model_parameters[game.home_team]
        away = self.model_parameters[game.away_team]
        h_a_factor = self.model_parameters["h_a_factor"]
        # Calculate home_team xg
        home_xg = intercept * home["att"] * away["def"] * h_a_factor
        # Calculate away_team xg
        away_xg = intercept * away["att"] * home["def"]
        # derive win probabilities
        prob_home_win = skellam.sf(0, home_xg, away_xg)
        prob_draw = skellam.pmf(0, home_xg, away_xg)
        prob_away_win = skellam.cdf(-1, home_xg, away_xg)
        # return report
        return Game_Prediction(game, home_xg, away_xg, prob_home_win, prob_away_win, prob_draw)
    
    # Returns simulation of specified game as a Game_Simulation object
        # Simulated is derived as random realization of goal counts from both home and away poisson distributions
    def Simulate_Game(self, game : Game):
        # Obtain game prediction
        game_prediction = self.Predict_Game(game)
        # Random-simulate home goals
        home_goals = poisson.rvs(game_prediction.home_pred_goals)
        # Random-simulate away goals
        away_goals = poisson.rvs(game_prediction.away_pred_goals)
        # Return [home, away] score
        return Game_Simulation(game, home_goals, away_goals)