import pandas as pd

def load_data(file_path):
    dataframe = pd.read_csv(file_path)
    dataframe["words"] = dataframe["words"].apply(split_data)
    dataframe["lid"] = dataframe["lid"].apply(split_data)
    return dataframe[["words", "lid"]].to_dict(orient="records")

def split_data(data):
    # Parses a string of the form "['a' 'b' 'c']" into a list of strings
    return [word.strip("'") for word in data.strip("[]").split()]

def load_data_split():
    return {
        "test": load_data("data/lid_spaeng_test.csv"),
        "train": load_data("data/lid_spaeng_train.csv"),
        "validation": load_data("data/lid_spaeng_validation.csv"),
    }