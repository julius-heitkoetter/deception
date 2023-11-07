import json
import os

def save_json_locally(storage_path, filename, data):
    """
    saves a json file locally

    Inputs:
        storage_path: path where data is to be stored
        filename: name of data that needs to be stored
        data: jsonable data (nested lists and dictionaries)
    """

    with open(os.path.join(storage_path, filename), 'w') as f:
        json.dump(data, f)

def get_json_locally(storage_path, filename):
    """
    gets a json file from local filesystem

    Inputs:
        storage_path: path where data is to be stored
        filename: name of data that needs to be stored
    """

    with open(os.path.join(storage_path, filename), 'r') as f:
        pairs = json.load(f)

    return pairs