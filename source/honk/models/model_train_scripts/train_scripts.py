# for defining an operations folder for storing model train scripts
from goose.operation.built_in_operations.goose_operations import goose_operations

# operations folder for storing model training scripts
# goose train
model_train_operations = goose_operations.create_subfolder("train", "train models")