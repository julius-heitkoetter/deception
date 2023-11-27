import typing as T
import json
import os
from pathlib import Path
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, HfApi
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


def filename_from_atoms(
        dataset: str, category: str, stage: str, deciever_model: T.Optional[str] = "anyModel", 
        supervisor_model: T.Optional[str] = "anyModel", timestamp: T.Optional[str] = None
        )-> str:
    """
    Returns a dataset filename from information like dataset, category, and stage along the chain
    of qa, qae, qaev, qaeve. Information is underscore delimited. Timestamped to the microsecond.

    Additionally, if deciever model or supervisor model are yet to be specified, it is considered
    to be "anyModel". 

    Example usage:
        filename_from_atoms(
            dataset="mmlu",
            category="econometrics",
            "stage"="qae",
        )
    """
    
    if any("_" in s for s in [dataset, category, stage, deciever_model, supervisor_model]):
        raise ValueError("Don't use underscores in dataset, category, or stage for filename.")
    
    if deciever_model=="anyModel" and (stage=="qae" or stage == "qaev" or stage == "qaeve"):
        raise ValueError("qae, qaev, and qaeve datasets must specify 'deciever_model'")
    
    if supervisor_model=="anyModel" and (stage == "qaev" or stage == "qaeve"):
        raise ValueError("qaev and qaeve datasets must specify 'supervisor_model'")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") if timestamp is None else timestamp
    return f"{dataset}_{category}_{stage}_{deciever_model}_{supervisor_model}_{timestamp}.json"


def atoms_from_filename(filename: str) -> T.Tuple[str, str, str, str, str, str]:
    """
    Returns the dataset, category, stage, deciever model, supervisor model, and timestamp of the filename.
    """

    path = Path(filename)
    if path.suffix != ".json":
        raise ValueError("Must use json file.")

    atoms = path.stem.split("_")

    if len(atoms) != 6:
        raise ValueError(
            "Expected 6 atoms in the filename (dataset, category, stage, deciever model, supervisor model, timestamp),"
            f"but found {len(atoms)}. Atoms are underscore delimited."
        )

    return tuple(atoms)


def next_filename_in_chain(filename: str, deciever_model: T.Optional[str] = None, supervisor_model: T.Optional[str] = None) -> str:
    """
    Returns the full filename of the next stage of the qa, qae, qaev, qaeve chain.

    Uses the same timestamp as the previous stage to keep same datasets organized.
    """

    next_stage_dict = {"qa": "qae", "qae": "qaev", "qaev": "qaeve"}

    # In states qae and qaev, we overwrite the model name with the model name that was passed into the function
    # In all other cases, we grab the same model used at before
    # If we're at a model that hasn't been specified, it should be named 'anyModel'.
    stage = atoms_from_filename(filename)[2]
    if stage == "qa":
        dataset, category, stage, _, supervisor_model_name, timestamp = atoms_from_filename(filename)
        if deciever_model is None:
            raise ValueError("Deciever model must be specified when going from qa to qae")
        deceiver_model_name = deciever_model
    elif stage == "qae":
        dataset, category, stage, deceiver_model_name, _, timestamp = atoms_from_filename(filename)
        if supervisor_model is None:
            raise ValueError("Supervisor model must be specified when going from qae to qaev")
        supervisor_model_name = supervisor_model
    else:
        dataset, category, stage, deceiver_model_name, supervisor_model_name, timestamp = atoms_from_filename(filename)

    

    if stage not in next_stage_dict:
        raise ValueError(f"Stage not recognized: {stage}. Known stages are {next_stage_dict.keys()}. Note that qaeve is the final stage and cannot be chained further.")

    next_stage = next_stage_dict[stage]

    return filename_from_atoms(dataset, category, next_stage, deceiver_model_name, supervisor_model_name, timestamp)

# The helper functions below form an interface to extract information out of dataset items.

def str_to_bool(string: str) -> bool:
    if type(string) == bool:
        return string
    # remove non-alphabetic characters, such as spaces and numbers
    cleaned_string = "".join(char for char in string if char.isalpha())
    return cleaned_string == "true"

def get_question(item: dict) -> str:
    return item["question"]

def get_answer(item: dict) -> str:
    return item["answer"]

def get_explanation(item: dict) -> str:
    return item["explanation"]

def get_correct(item: dict) -> bool:
    return str_to_bool(item["correct"])

def get_verdict(item: dict) -> bool:
    return str_to_bool(item["verdict"])

def get_deceiver_evaluation(item: dict) -> bool:
    return str_to_bool(item["evaluation-deceiver"])

def get_supervisor_evaluation(item: dict) -> bool:
    return str_to_bool(item["evaluation-supervisor"])
