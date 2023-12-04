import json
import sys
import os
import platform
import argparse

from lib.models import *
from config import *
from lib.chain import Supervisor, Deceiver, Evaluator
from lib.utils import save_json_locally, get_json_locally, upload_json_to_hf
from lib.dataset import MMLULoader, EthicsLoader

def run_pipeline_on_dataset(
        dataset_name,             # Must either be 'mmlu' or 'ethics'
        category,                 # Must be in the categories of the respective dataset
        save_location,            # Must be either 'local' or 'hf'
        deceiver_model_name,      # Must be in the keys to MODEL_MAPPING
        deceiver_config_name,     # Must be in the keys to CONFIG_MAPPING
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
        "llama_7b_base_config" : llama_7b_base_config,
        "llama_13b_base_config" : llama_13b_base_config,
        "llama_70b_base_config" : llama_70b_base_config,
        "llama_7b_noRLHF_config" : llama_7b_noRLHF_config,
        "llama_13b_noRLHF_config" : llama_13b_noRLHF_config,
        "llama_70b_noRLHF_config" : llama_70b_noRLHF_config,
    }

    # check if models and configs are well defined
    if deceiver_model_name not in MODEL_MAPPING.keys():
        raise ValueError("Deceiver model not in known models")
    if deceiver_config_name not in CONFIG_MAPPING.keys():
        raise ValueError("Deceiver config not in known configurations")
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

    # create language models
    deceiver_llm = MODEL_MAPPING[deceiver_model_name](**CONFIG_MAPPING[deceiver_config_name])
    supervisor_llms = [MODEL_MAPPING[supervisor_model_name](**CONFIG_MAPPING[supervisor_config_name])
                     for supervisor_model_name, supervisor_config_name in zip(supervisor_model_names, supervisor_config_names)]

    # create supervisor, deceiver, and evaluator
    deceiver = Deceiver(deceiver_llm)
    supervisors = [Supervisor(supervisor_llm) for supervisor_llm in supervisor_llms]
    evaluators = [Evaluator(supervisor_llm) for supervisor_llm in supervisor_llms]

    # create dataloader
    if dataset_name == 'mmlu':
        dataloader = MMLULoader()
    elif dataset_name == 'ethics':
        dataloader = EthicsLoader()
    else:
        assert ValueError("dataset_name must either be 'mmlu' or 'ethics'")
    
    # create qa datasets
    print("INFO: Starting qa dataset generation")
    qa_correct_dataset_path, qa_incorrect_dataset_path = dataloader(category, data_split="test", save_locally=save_locally, save_on_hf=save_on_hf, num_samples=num_samples)
    print("INFO: Finsihed qa correct dataset generation. File at:", save_location, ":", qa_correct_dataset_path)
    print("INFO: Finished qa incorrect dataset generation, File at:", save_location, ":", qa_incorrect_dataset_path)
    
    # process through the incorrect dataset (main work is done below)
    print("INFO: Starting qae incorrect dataset generation")
    qae_incorrect_dataset_path = deceiver.run_on_dataset_name(qa_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qae incorrect dataset generation. File at:", save_location, ":", qae_incorrect_dataset_path)
    for supervisor, evaluator, i in zip(supervisors, evaluators, range(len(supervisors))):
        print("INFO: Starting qaev incorrect dataset generation for supervisor number:", i)
        qaev_incorrect_dataset_path = supervisor.run_on_dataset_name(qae_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaev incorrect dataset generation. File at:", save_location, ":", qaev_incorrect_dataset_path)
        print("INFO: Starting qaeve incorrect dataset generation for evaluator number:", i)
        qaeve_incorrect_dataset_path = evaluator.run_on_dataset_name(qaev_incorrect_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaeve incorrect dataset generation. File at:", save_location, ":", qaeve_incorrect_dataset_path)
   
    # process through the correct dataset (main work is done below)
    print("INFO: Starting qae correct dataset generation")
    qae_correct_dataset_path = deceiver.run_on_dataset_name(qa_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
    print("INFO: Finished qae correct dataset generation. File at:", save_location, ":", qae_correct_dataset_path)
    for supervisor, evaluator, i in zip(supervisors, evaluators, range(len(supervisors))):
        print("INFO: Starting qaev correct dataset generation for supervisor number:", i)
        qaev_correct_dataset_path = supervisor.run_on_dataset_name(qae_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaev correct dataset generation. File at:", save_location, ":", qaev_correct_dataset_path)
        print("INFO: Starting qaeve correct dataset generation for supervisor number:", i)
        qaeve_correct_dataset_path = evaluator.run_on_dataset_name(qaev_correct_dataset_path, save_locally=save_locally, save_on_hf=save_on_hf)
        print("INFO: Finished qaeve correct dataset generation. File at:", save_location, ":", qaeve_correct_dataset_path) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run entire pipeline with multiple supervisors to generate qaeve dataset (intermediate steps saved along the way)')

    def list_of_strings(arg):
        return arg.split(',')

    # Mandatory arguments
    parser.add_argument('dataset_name', type=str, choices=['mmlu', 'ethics'], help='Dataset name')
    parser.add_argument('category', type=str, help='Category of the dataset')
    parser.add_argument('save_location', type=str, choices=['local', 'hf'], help='Save location')
    parser.add_argument('deceiver_model_name', type=str, help='Deceiver model name')
    parser.add_argument('deceiver_config_name', type=str, help='Deceiver config name')
    parser.add_argument('supervisor_model_names', type=list_of_strings, help='List of Supervisor model names')
    parser.add_argument('supervisor_config_names', type=list_of_strings, help='List of Supervisor config names')

    # Optional arguments
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (optional)')

    args = parser.parse_args()

    run_pipeline_on_dataset(args.dataset_name, args.category, args.save_location, args.deceiver_model_name, args.deceiver_config_name, args.supervisor_model_names, args.supervisor_config_names, args.num_samples)
