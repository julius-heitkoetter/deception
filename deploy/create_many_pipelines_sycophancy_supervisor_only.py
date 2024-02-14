import copy
import os
from lib.utils import atoms_from_filename
import json

template = """#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --time=36:00:00 
#SBATCH --partition=single
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --job-name=sycophancy-{category}-{deceiver_model_name}
#SBATCH --output={logging_path}/{logging_file}

echo Running Job: {category}, {save_location}, {deceiver_model_name}, {supervisor_configs}, {coefficients}
echo Starting Setup
source ~/.bashrc
cd {working_directory}
conda activate correlated_errors 
source install.sh 
source setup.sh
echo Setup Complete

echo Starting Pipeline Run
python bin/dataset_pipeline_sycophancy_multiple_supervisors.py {qae_correct_dataset_path} {qae_incorrect_dataset_path} {save_location} {supervisor_models} {supervisor_configs} {steering_vector_paths} {coefficients}
"""

config_to_model_name = {
    'llama_7b_base_config': 'llama_7b',
    'llama_13b_base_config': 'llama_13b',
    'llama_70b_base_config': 'llama_70b',
}

CONFIG_MAPPING_TO_NAME = {
    "gpt_35_turbo_base_config":"gpt-3.5-turbo",
    "gpt_4_base_config":"gpt-4-turbo",
    "llama_7b_base_config" : "Llama-2-7b-chat-hf",
    "llama_13b_base_config" : "Llama-2-13b-chat-hf",
    "llama_70b_base_config" : "Llama-2-70b-chat-hf",
}

MODEL_CONFIG_TO_STEERING_VECTOR = {
    'llama_7b_base_config': {
        15: '/data/misha_gerovitch/correlated_llm_errors/steering-vectors/vec_layer_15_Llama-2-7b-chat-hf.pt',
    },
    'llama_13b_base_config': {
        15: '/data/misha_gerovitch/correlated_llm_errors/steering-vectors/vec_layer_15_Llama-2-13b-chat-hf.pt',
    }
}

def find_filename_pair(category, deceiver_model, directory_path):
    file_path_pairs = []
    for filename in os.listdir(directory_path):
        if f"{category}_qae_{deceiver_model}" in filename:
            file_path_pairs.append(os.path.join(directory_path,filename))
    assert len(file_path_pairs) == 2, "There should be exactly two files for each category and deceiver model"
    
    is_first_correct = True
    with open(file_path_pairs[0], 'r') as file:
        data = json.load(file)
        is_first_correct = data['metadata']['correct']
    
    # separate local file path prefix to leave hf file path
    file_path_pairs[0] = '/'.join(os.path.join(directory_path,file_path_pairs[0]).split('/')[4:])
    file_path_pairs[1] = '/'.join(os.path.join(directory_path,file_path_pairs[1]).split('/')[4:])

    if is_first_correct:
        return file_path_pairs[0], file_path_pairs[1]
    else:
        return file_path_pairs[1], file_path_pairs[0]

def generate_script(config):
    supervisor_models = ','.join(config['supervisor_models'])
    supervisor_configs = ','.join(config['supervisor_configs'])
    coefficients = ','.join(config['coefficients'])
    steering_vector_path = ','.join(config['steering_vector_paths'])

    num_gpus = 0
    if '7b' in supervisor_configs or '13b' in supervisor_configs:
        num_gpus = 1
    if '70b' in supervisor_configs:
        num_gpus = 2

    qae_correct_dataset_path, qae_incorrect_dataset_path = find_filename_pair(
        config['category'], 
        config['deceiver_model_name'], 
        config['dataset_directory']
    )
    
    script = template.format(
        qae_correct_dataset_path=qae_correct_dataset_path,
        qae_incorrect_dataset_path=qae_incorrect_dataset_path,
        logging_path=config['logging_path'],
        logging_file=config['logging_file'],
        working_directory=config['working_directory'],
        save_location=config['save_location'],
        supervisor_models=supervisor_models,
        supervisor_configs=supervisor_configs,
        steering_vector_paths=steering_vector_path,
        category=config['category'],
        deceiver_model_name=config['deceiver_model_name'],
        coefficients=coefficients,
        num_gpus= num_gpus,
    )
    return script

def generate_multiple_scripts(config):
    categories = config['categories']

    deceiver_configs = config['deceiver_configs']

    config['steering_vector_paths'] = [
        MODEL_CONFIG_TO_STEERING_VECTOR[supervisor_config][15] for supervisor_config in config['supervisor_configs']
    ]

    scripts = []

    for category in categories:
        for deceiver_config in deceiver_configs:

            script_config = copy.deepcopy(config)
            script_config['deceiver_model_name'] = CONFIG_MAPPING_TO_NAME[deceiver_config]
            script_config['deceiver_config'] = deceiver_config
            script_config['category'] = category
            del script_config['deceiver_configs']
            del script_config['categories']

            logging_file_name = script_config['category']+ "-" + deceiver_config + "-%j.log"
            script_config['logging_file'] = logging_file_name

            script = generate_script(script_config)

            script_file_name = f"sycophancy-job-{script_config['category']}-{deceiver_config}.sbatch"
            script_save_path = os.path.join(script_config['script_saving_path'], script_config['deceiver_config'], script_file_name)
            if not os.path.exists(os.path.join(script_config['script_saving_path'], script_config['deceiver_config'])):
                os.makedirs(os.path.join(script_config['script_saving_path'], script_config['deceiver_config']))
            with open(script_save_path, 'w') as file:
                file.write(script)


    # for qae_correct_dataset_path, qae_incorrect_dataset_path in filename_pairs:
    #     script_config = copy.deepcopy(config)

    #     script_config['qae_correct_dataset_path'] = qae_correct_dataset_path
    #     script_config['qae_incorrect_dataset_path'] = qae_incorrect_dataset_path
        
    #     for supervisor_model, supervisor_config, steering_vector_path in zip(supervisor_models, supervisor_configs, steering_vector_paths):
    #         supervisor_model_name = config_to_model_name[supervisor_config]
    #         dataset, category, stage, deceiver_model_name, _, timestamp = atoms_from_filename(qae_correct_dataset_path)
    #         pair_dataset, pair_category, pair_stage, pair_deceiver_name, _, _ = atoms_from_filename(qae_incorrect_dataset_path)

    #         assert dataset == pair_dataset, "Datasets must match."
    #         assert category == pair_category, "Categories must match."
    #         assert stage == pair_stage, "Stages must match."
    #         assert deceiver_model_name == pair_deceiver_name, "Deceiver models must match."
    #         assert stage == "qae", "Stage must be qae for sycophancy experiments."


    #         script_config['supervisor_model'] = supervisor_model
    #         script_config['supervisor_config'] = supervisor_config
    #         script_config['steering_vector_path'] = steering_vector_path

    #         run_name = f"sycophancy_{dataset}_{category}_{deceiver_model_name}_{supervisor_model_name}"
    #         script_config['run_name'] = run_name

    #         logging_file_name = run_name + "-%j.log"
            
    #         script_config["experiment_name"] = logging_file_name
    #         script_config['logging_file'] = logging_file_name

    #         script = generate_script(script_config)

    #         script_file_name = "run-job-" + run_name + ".sbatch"
    #         script_save_path = os.path.join(script_config['script_saving_path'], run_name, script_file_name)
    #         if not os.path.exists(os.path.join(script_config['script_saving_path'], run_name)):
    #             os.makedirs(os.path.join(script_config['script_saving_path'], run_name))
    #         with open(script_save_path, 'w') as file:
    #             file.write(script)


# Example usage
llama_sycophancy_config = {
    'logging_path': '/data/misha_gerovitch/correlated_llm_errors/deploy/logging_sycophancy',
    'script_saving_path': '/data/misha_gerovitch/correlated_llm_errors/deploy/batch_scripts_sycophancy',
    'working_directory': '/data/misha_gerovitch/correlated_llm_errors',
    'dataset_directory': '/data/misha_gerovitch/correlated-errors/data/sycophancy-2024-02-13',
    'save_location': 'hf',
    'categories': ["high-school-computer-science", "elementary-mathematics", "high-school-psychology", "high-school-statistics"],
    'deceiver_configs': ['llama_70b_base_config'],
    'supervisor_models': ['LlamaLLM', 'LlamaLLM'],
    'supervisor_configs': ['llama_7b_base_config', 'llama_13b_base_config'],
    'coefficients': ['f-1.0', 'f1.0'],
}


if __name__ == "__main__":

    generate_multiple_scripts(llama_sycophancy_config)
    print("Done!")
