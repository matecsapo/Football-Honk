# for defining an operations folder for storing model evaluation operations
from goose.operation.built_in_operations.goose_operations import goose_operations

# operations folder for storing model evaluation operations
# goose evaluate
model_evaluation_operations = goose_operations.create_subfolder("evaluate", "evaluate model")

# operation for evaluating a model on MAE
# goose train league-sprm [league]
@model_train_operations.operation("league-sprm", "train sprm model for specified league")
def train_league_sprm(league : str):
    # standardize league name
    if isinstance(league, str):
        league = League(league)
    # build model for specified league based on most recent data
    model_name = league.league + "_sprm"
    print(f"Training {model_name}...")
    model = Static_Poi_Reg_Model(model_name)
    model.Add_Data(results_data.Retrieve(league, Season(2026)))
    model.Process_Data()
    model.Split_Train_Test(datetime.now())
    model.Train_Model()
    # save model to honk/models/active_models/
    model.save_model_fgm(Path(__file__).parent.parent)
    print(f"Trained and saved {model_name}")