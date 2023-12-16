import json
import sys
import os
import platform
import argparse
import gc
import torch

from lib.models import *
from config import *
from lib.chain import Supervisor, Deceiver, Evaluator
from lib.utils import save_json_locally, get_json_locally, upload_json_to_hf
from lib.dataset import MMLULoader, EthicsLoader

def run_pipeline_on_dataset(
        qae_correct_dataset_path,  # Path for correct dataset
        qae_incorrect_dataset_path,# Path for incorrect dataset
        save_location,            # Must be either 'local' or 'hf'
        supervisor_model_names,   # Must be a list with strings contained in the keys to MODEL_MAPPING
        supervisor_config_names,  # Must be a list with strings contained in the keys to CONFIG_MAPPING
        num_samples = None,       # Optional integer argument, containing the number of samples
    ):

    MODEL_MAPPING = {
        "OpenAILLM":OpenAILLM, 
        "LlamaLLM": LlamaLLM
    }
    CONFIG_MAPPING = {
        "gpt_35_turbo_base_config":gpt_35_turbo_base_config,
        "gpt_4_base_config":gpt_4_base_config,
        "llama_7b_base_config" : llama_7b_base_config,
        "llama_13b_base_config" : llama_13b_base_config,
        "llama_70b_base_config" : llama_70b_base_config,
        "llama_7b_noRLHF_config" : llama_7b_noRLHF_config,
        "llama_13b_noRLHF_config" : llama_13b_noRLHF_config,
        "llama_70b_noRLHF_config" : llama_70b_noRLHF_config,
    }

    print("Supervisor model names : ", supervisor_model_names)
    print("Supervisor model configs: ", supervisor_config_names)
    for supervisor_model_name, supervisor_config_name in zip(supervisor_model_names, supervisor_config_names):
        if supervisor_model_name not in MODEL_MAPPING.keys():
            raise ValueError("Supervisor model not in known models")
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

    for supervisor_model_name, supervisor_config_name in zip(supervisor_model_names, supervisor_config_names):
        
        #create models
        supervisor_llm = MODEL_MAPPING[supervisor_model_name](**CONFIG_MAPPING[supervisor_config_name])
        supervisor = Supervisor(supervisor_llm)
        evaluator = Evaluator(supervisor_llm)
        
        # run validation and evaluation on incorrect dataset
        print("INFO: Starting qaev incorrect dataset generation for supervisor:", supervisor_config_name)
        qaev_incorrect_dataset_path = supervisor.run_on_dataset_name(qae_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaev incorrect dataset generation. File at:", save_location, ":", qaev_incorrect_dataset_path)
        print("INFO: Starting qaeve incorrect dataset generation for evaluator:", supervisor_config_name)
        qaeve_incorrect_dataset_path = evaluator.run_on_dataset_name(qaev_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaeve incorrect dataset generation. File at:", save_location, ":", qaeve_incorrect_dataset_path)
        
        # run validation and evaluation on correct I'm dataset
        print("INFO: Starting qaev correct dataset generation for supervisor:", supervisor_config_name)
        qaev_correct_dataset_path = supervisor.run_on_dataset_name(qae_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaev correct dataset generation. File at:", save_location, ":", qaev_correct_dataset_path)
        print("INFO: Starting qaeve correct dataset generation for supervisor:", supervisor_config_name)
        qaeve_correct_dataset_path = evaluator.run_on_dataset_name(qaev_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaeve correct dataset generation. File at:", save_location, ":", qaeve_correct_dataset_path) 
        
        # delete models to make room on GPU
        del supervisor
        del evaluator
        del supervisor_llm
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entire pipeline with multiple supervisors to generate qaeve dataset starting at a qae dataset.')

    def list_of_strings(arg):
        return arg.split(',')

    # Mandatory arguments
    parser.add_argument('qae_correct_dataset_path', type=str, help='Correct QAE dataset path')
    parser.add_argument('qae_incorrect_dataset_path', type=str, help='Incorrect QAE dataset path')
    parser.add_argument('save_location', type=str, choices=['local', 'hf'], help='Save location')
    parser.add_argument('supervisor_model_names', type=list_of_strings, help='List of Supervisor model names')
    parser.add_argument('supervisor_config_names', type=list_of_strings, help='List of Supervisor config names')

    # Optional arguments
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (optional)')

    args = parser.parse_args()

    run_pipeline_on_dataset(args.qae_correct_dataset_path, args.qae_incorrect_dataset_path, args.save_location, args.supervisor_model_names, args.supervisor_config_names, args.num_samples)
