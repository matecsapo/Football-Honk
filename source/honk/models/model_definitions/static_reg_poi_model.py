# for implementing a model
from goose.model import Model

# For data manipulation
from goose.data.goose_data_structures import Game, Game_Prediction
from goose.data.pull_data import Results_Data
import numpy as np
import pandas as pd
import json as json
import os as os
from pathlib import Path
from typing import Self

# For fitting poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf

# For evaluating model effectiveness
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# For forecasting results of games
from scipy.stats import poisson, skellam

# Class for static poison regression model
# Builds a poisson regression model on XG as an explanation of 41 parameters:
    # att, def values for each 20 teams
    # league-consistent estiamte of home advantage factor
# Data can be added to model via Add_Data()
# Training and testing data specified via Set_Train_Data() and Set_Test_Data()
@Model.define_model("Static Poisson Regression Model")
class Static_Poi_Reg_Model(Model):
    def __init__(self, model_name):
        self.Model_Name = model_name
        self.Data = None # Results_Data object
        self.Processed_Data = None # Regular pd dataframe
        self.Train_Data = None # Regular pd dataframe
        self.Test_Data = None # Regular pd dataframe
        self.Model = None
        self.Model_Parameters = None # this is a struct of all the actual "unconverted" values
        self.Model_Evals = None # pd dataframe containing evaluation statistics of model

    # Add data for model training/testing
    def Add_Data(self, data : Results_Data):
        self.Data = data

    # Processes train+/test data into:
        # produced xg | team, opponent, h/a
    # Date is maintained for purposes of Split_Train_Test
    def Process_Data(self):
        # columns to keep for this model
        cols = ["date", "home_team", "away_team", "home_xg", "away_xg"]
        self.Processed_Data = self.Data.data[cols].reset_index(drop=True)
        # Convert date column to datetime objects
        self.Processed_Data["date"] = pd.to_datetime(self.Processed_Data["date"])
        # Duplicate each game = datapoint from each team's perspective
        home_perspective = pd.DataFrame(self.Processed_Data[["date", "home_team", "away_team", "home_xg"]])
        home_perspective["h_a"] = 1 # true
        home_perspective = home_perspective.rename(columns = {"home_team" : "team", "away_team" : "opponent", "home_xg" : "xg"})
        away_perspective = pd.DataFrame(self.Processed_Data[["date", "away_team", "home_team", "away_xg"]])
        away_perspective["h_a"] = 0 # false
        away_perspective = away_perspective.rename(columns = {"away_team" : "team", "home_team" : "opponent", "away_xg" : "xg"})
        # produced xg | team, opponent, h/a    
        self.Processed_Data = pd.concat([home_perspective, away_perspective], ignore_index=True)
        self.Processed_Data = self.Processed_Data[["date", "team", "opponent", "h_a", "xg"]]
    
    # Splits processed_data into train and test data according to specified cutoff date
        # Sets training data as from <= a certain date
        # Sets test data as > a certain date
    def Split_Train_Test(self, date):
        self.Train_Data = self.Processed_Data[self.Processed_Data["date"] <= date]
        self.Test_Data = self.Processed_Data[self.Processed_Data["date"] > date]

    # Trains model to predict goals scored, G, by team in [team, opponent, h/a] fixture
    # G ~ Poi(g), g = att_team * def_opponent * h_a_factor,
    # parameters att_team, def_opponent for all 20 teams + h_a_factor estimated via generalized linear regression on:
        # ln(g) ~ ln(team1_att) + ln(team2_def) + ... + ln(teamn_att) + ln(teamn_def) + ln(h_a_factor)
    def Train_Model(self):
        # train model
        self.Model = self.Model = smf.glm(  
            formula="xg ~ team + opponent + h_a", # formula automatically expands categorial home/team columns into one-for-each of 20 teams
            data = self.Train_Data,
            family=sm.families.Poisson() # poisson regression
        ).fit()
        # Extract parameters from model
        params = self.Model.params
        # Exponentiate out of log-space
        params = np.exp(params)
        # store parameters into "nice" struct
        self.Model_Parameters = {}
        # Intercept
        self.Model_Parameters["Intercept"] = params["Intercept"]
        # (all) Team att and def values
        for team in sorted(self.Train_Data['team'].unique()):
            self.Model_Parameters[team] = {"att" : params.get(f"team[T.{team}]", 1), # naming structure employed by smf
                                             "def" : params.get(f"opponent[T.{team}]", 1)} # " "
        # h/a_factor
        self.Model_Parameters["h_a_factor"] = params["h_a"]
    
    # Tests Model incl:
        # mean difference
        # mean squared difference
    def Test_Model(self):
        # actual values of G for Train_Data
        actual_ln_goals = self.Test_Data["xg"]
        # Predicted values by self.Model of G for Train_Data
        predicted_ln_goals = self.Model.predict(self.Test_Data)
        # Various tests...
        self.Model_Evals = {
            "Mean Squared Error" : mean_squared_error(actual_ln_goals, predicted_ln_goals),
            "Root Mean Squared Error" : root_mean_squared_error(actual_ln_goals, predicted_ln_goals)
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
        self.Model.save(folder / "smfmodel.pkl")
        # json Model_Paramaters dump
        with open(folder / "parameters.json", "w") as f:
            json.dump(self.Model_Parameters, f, indent=4)
        # model evaluation statistics
        with open(folder / "eval_statistics.json", "w") as f:
            json.dump(self.Model_Evals, f, indent=4)

    # Loads model saved to folder self.Model_Name/ given path
    @classmethod
    def load_model(cls, model_save_path : str | Path) -> Self:
        model_name = Path(model_save_path).name
        model = cls(model_name)
        # load smf dump
        model.Model = sm.load(model_save_path / "smfmodel.pkl")
        # load parameters
        with open(model_save_path / "parameters.json", "r") as f:
            model.Model_Parameters = json.load(f)
        # return loaded model
        return model

    # Produces a prediction report for a given supplied game
    # Returns a Game_Prediction report
    # home win, away win, draw probabilities found via skellam distribution comparing home and away dists
    def Predict_Game(self, game : Game):
        # Extract all necessary model ratings
        intercept = self.Model_Parameters["Intercept"]
        home = self.Model_Parameters[game.home_team]
        away = self.Model_Parameters[game.away_team]
        h_a_factor = self.Model_Parameters["h_a_factor"]
        # Predict home_team xg
        home_xg = intercept * home["att"] * away["def"] * h_a_factor
        # Predict away_team xg
        away_xg = intercept * away["att"] * home["def"]
        # Predict win probabilities
        prob_home_win = skellam.sf(0, home_xg, away_xg)
        prob_draw = skellam.pmf(0, home_xg, away_xg)
        prob_away_win = skellam.cdf(-1, home_xg, away_xg)
        # return report
        return Game_Prediction(game, home_xg, away_xg, prob_home_win, prob_away_win, prob_draw)
    
    # Randomly simulates a result for specified game according to model prediction
    def Simulate_Game(self, game : Game):
        # Obtain game prediction
        game_prediction = self.Predict_Game(game)
        # Random-simulate home goals
        home_goals = poisson.rvs(game_prediction.home_xg)
        # Random-simulate away goals
        away_goals = poisson.rvs(game_prediction.away_xg)
        # Return [home, away] score
        return home_goals, away_goals