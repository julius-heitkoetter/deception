import json
import sys
import os
import platform

from lib.models import OpenAILLM, LlamaLLM
from config import gpt_35_turbo_base_config, llama_base_config, llama_70b_base_config
from lib.chain import Supervisor, Deceiver, Evaluator
from lib.utils import save_json_locally, get_json_locally, upload_json_to_hf
from lib.dataset import MMLULoader, EthicsLoader

def run_pipeline_on_dataset(
        dataset_name,             # Must either be 'mmlu' or 'ethics'
        category,                 # Must be in the categories of the respective dataset
    save_location,            # Must be either 'local' or 'hf'
        deceiver_model_name,      # Must be in the keys to MODEL_MAPPING
        deceiver_config_name,     # Must be in the keys to CONFIG_MAPPING
        supervisor_model_name,    # Must be in the keys to MODEL_MAPPING
        supervisor_config_name,   # Must be in the keys to CONFIG_MAPPING
        num_samples = None,       # Optional integer argument, containing the number of samples
    ):

    print("HERE")

    MODEL_MAPPING = {
        "OpenAILLM":OpenAILLM, 
        "LlamaLLM": LlamaLLM
    }
    CONFIG_MAPPING = {
        "gpt_35_turbo_base_config":gpt_35_turbo_base_config,
        "llama_base_config" : llama_base_config,
        "llama_70b_base_config" : llama_70b_base_config,
    }

    # check if models and configs are well defined
    if deceiver_model_name not in MODEL_MAPPING.keys():
        raise ValueError("Deceiver model not in known models")
    if supervisor_model_name not in MODEL_MAPPING.keys():
        raise ValueError("Supervisor model not in known models")
    if deceiver_config_name not in CONFIG_MAPPING.keys():
        raise ValueError("Deceiver config not in known configurations")
    if supervisor_config_name not in CONFIG_MAPPING.keys():
        raise ValueError("Supervisor config not in known configurations")
    
    if save_location=='local':
        save_locally = True
        save_on_hf = False
    elif save_location=='hf':
        save_locally = False
        save_on_hf = True
    else:
        raise ValueError("Save location must either be 'local' or 'hf'")

    # create language models
    deceiver_llm = MODEL_MAPPING[deceiver_model_name](**CONFIG_MAPPING[deceiver_config_name])
    supervisor_llm = MODEL_MAPPING[supervisor_model_name](**CONFIG_MAPPING[supervisor_config_name])

    # create supervisor, deceiver, and evaluator
    deceiver = Deceiver(deceiver_llm)
    supervisor = Supervisor(supervisor_llm)
    evaluator = Evaluator(supervisor_llm)

    # create dataloader
    if dataset_name == 'mmlu':
        dataloader = MMLULoader()
    elif dataset_name == 'ethics':
        dataloader = EthicsLoader()
    else:
        assert ValueError("dataset_name must either be 'mmlu' or 'ethics'")
    
    # create qa datasets
    qa_correct_dataset_path, qa_incorrect_dataset_path = dataloader(category, save_locally=save_locally, save_on_hf=save_on_hf)

    # process through the incorrect dataset (main work is done below)
    print("INFO: Starting qae incorrect dataset generation")
    qae_incorrect_dataset_path = deceiver.run_on_dataset_name(qa_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qae incorrect dataset generation. File at:", save_location, ":", qae_incorrect_dataset_path)
    print("INFO: Starting qaev incorrect dataset generation")
    qaev_incorrect_dataset_path = supervisor.run_on_datset_name(qae_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qaev incorrect dataset generation. File at:", save_location, ":", qaev_incorrect_dataset_path)
    print("INFO: Starting qaeve incorrect dataset generation")
    qaeve_incorrect_dataset_path = evaluator.run_on_dataset_name(qaev_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qaeve incorrect dataset generation. File at:", save_location, ":", qaeve_incorrect_dataset_path)
   
    # process through the correct dataset (main work is done below)
    print("INFO: Starting qae correct dataset generation")
    qae_correct_dataset_path = deceiver.run_on_dataset_name(qa_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qae correct dataset generation. File at:", save_location, ":", qae_correct_dataset_path)
    print("INFO: Starting qaev correct dataset generation")
    qaev_correct_dataset_path = supervisor.run_on_datset_name(qae_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qaev correct dataset generation. File at:", save_location, ":", qaev_correct_dataset_path)
    print("INFO: Starting qaeve correct dataset generation")
    qaeve_correct_dataset_path = evaluator.run_on_dataset_name(qaev_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qaeve correct dataset generation. File at:", save_location, ":", qaeve_correct_dataset_path) 

run_pipeline_on_dataset(
        'mmlu',
        'anatomy',
        'hf',
        'OpenAILLM',
        'gpt_35_turbo_base_config',
        'OpenAILLM',
        'gpt_35_turbo_base_config',
        5,    
)
