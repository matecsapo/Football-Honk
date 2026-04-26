# for defining a model training operation
from honk.models.model_train_scripts.train_scripts import model_train_operations

# for training sprm models
from goose.name_standardization import standardize_league_name
from honk.models.model_definitions.static_reg_poi_model import Static_Poi_Reg_Model
from goose.data.pull_data import Results_Data
from datetime import datetime
from pathlib import Path

# operation for training sprm model for specified league
# goose train sprm [league]
@model_train_operations.operation("sprm", "train sprm model for specified league")
def train_sprm(league : str):
    # standardize league name
    league = standardize_league_name(league)
    # build model for specified league based on most recent data
    model_name = league + "_sprm"
    print(f"Training {model_name}...")
    model = Static_Poi_Reg_Model(model_name)
    model.Add_Data(Results_Data(league, "2025/2026"))
    model.Process_Data()
    model.Split_Train_Test(datetime.now())
    model.Train_Model()
    # save model to honk/models/active_models/
    model.save_model_fgm(Path(__file__).parent.parent)
    print(f"Trained and saved {model_name}")