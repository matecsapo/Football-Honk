# for employing a model
from goose.model import Model

# for storing games
from goose.data.goose_data_structures.game_storage import Game, Completed_Game, Game_Prediction, Games
from collections import defaultdict

# function for conducting a walk-forward backtest of a stateful model over a specified set of games
# walk-forward traverses unique match-times chronologically:
    # produces Game_Prediction's using model for all remaining matches
    # .update()-ing model state with match-times games
# Returns a dictionary of mappings match-time datetime --> Games[Game_Prediction] container
# parameterized with:
        # model to use
        # set of games to backtest on
def Walk_Forward_Backtest(self, model : Model, backtest_games : Games[Completed_Game]):
    # dictionary storing produced Game_Predictions match-time datetime --> Games[Game_Prediction]
    game_predictions = {}
    # group backtest_games by unique match-times
    game_groups = defaultdict(list)
    for game in backtest_games.games:
        game_groups[game.date].append(game)
    # walk forward over all groups of games
    for match_time, games in game_groups.items():
        # produce predictions for remaining games
        remaining_games = [g for g in self.backtest_games.games if g.date >= match_time]
        remaining_game_predictions = Games[Game_Prediction]()
        for game in remaining_games:
            prediction = model.Predict_Game(game)
            remaining_game_predictions.Add_Game(prediction)
        # store predictions, indexed by matchtime, into self.game_predictions
        game_predictions[match_time] = remaining_game_predictions
        # .update() model state with group's games
        model.update(games)
    # return game_predictions
    return game_predictions
