import typing as T
import json
import os
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi
from datasets import load_dataset
import requests


def save_json_locally(storage_path, filename, data):
    """
    saves a json file locally

    Inputs:
        storage_path: path where data is to be stored
        filename: name of data that needs to be stored
        data: jsonable data (nested lists and dictionaries)
    """
    if storage_path and not os.path.exists(storage_path):
        os.makedirs(storage_path)
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


def upload_json_to_hf(obj: T.Any, path_in_repo: str, repo_id: str, repo_type: str):
    """
    Upload a json object to Hugging Face.

    Parameters:
        obj: object to upload to Hugging Face (can be json or arbitrary instance)
        path_in_repo: path to upload the file into, relative to Hugging Face repo root
        repo_id: which repo to upload to in Hugging Face
        repo_type: choose "model" or "dataset" or "space"

    Example usage:
        upload_json_to_hf(
            obj=dataset_dict,
            path_in_repo="dummy_dataset.json",
            repo_id="laker-julius-misha/correlated-errors",
            repo_type="dataset",
        )
    """

    # temporarily make a file to store the json object
    date_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_path = f"temp_hf_{date_id}.json"
    save_json_locally(storage_path="", filename=temp_path, data=obj)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    # remove the temporary json file
    os.remove(temp_path)


def download_json_dataset_from_hf(path_in_repo: str, repo_id: str) -> dict:
    """
    Returns file stored in Hugging Face dataset.

    Parameters:
        path_in_repo: path to download the json file from, relative to Hugging Face repo root
        repo_id: which repo to download from in Hugging Face (must be type "dataset")

    download_dataset_from_hf(
        path_in_repo="dummy_dataset.json",
        repo_id="laker-julius-misha/correlated_errors",
    )
    """

    if Path(path_in_repo).suffix != ".json":
        raise ValueError("Can only download json datasets from Hugging Face.")

    request_file = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"
    response = requests.get(request_file)
    return json.loads(response.content)


def download_pretrained_model_from_hf(model: str) -> T.Tuple[T.Any, T.Any]:
    """
    Returns (tokenizer, model) given a pretrained model available on Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    return tokenizer, model
