import copy
import os

template = """#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --time=36:00:00 
#SBATCH --partition=single
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --job-name={experiment_name}-{category}-{deceiver_config}
#SBATCH --output={logging_path}/{logging_file}

echo Running Job: {experiment_name}, {dataset}, {category}, {save_location}, {deceiver_model}, {deceiver_config}, {supervisor_models}
echo Starting Setup
source ~/.bashrc
cd {working_directory}
conda activate correlated_errors 
source install.sh 
source setup.sh
echo Setup Complete

echo Starting Pipeline Run
python bin/dataset_pipeline_multiple_supervisors.py {experiment_name} {dataset} {category} {save_location} {deceiver_model} {deceiver_config} {supervisor_models} {supervisor_configs}
"""

def generate_script(config):
    supervisor_models = ','.join(config['supervisor_models'])
    supervisor_configs = ','.join(config['supervisor_configs'])
    script = template.format(
        experiment_name=config['experiment_name'],
        logging_path=config['logging_path'],
        logging_file=config['logging_file'],
        working_directory=config['working_directory'],
        dataset=config['dataset'],
        category=config['category'],
        save_location=config['save_location'],
        deceiver_model=config['deceiver_model'],
        deceiver_config=config['deceiver_config'],
        supervisor_models=supervisor_models,
        supervisor_configs=supervisor_configs,
        num_gpus= 2 if ('70b' in supervisor_configs+config['deceiver_config']) else 1 #2 gpus needed for llama270b
    )
    return script

def generate_multiple_scripts(config):
    deceiver_models = config['deceiver_models']
    deceiver_configs = config['deceiver_configs']
    categories = config['categories']

    scripts = []

    for category in categories:
        for deceiver_model, deceiver_config in zip(deceiver_models, deceiver_configs):

            script_config = copy.deepcopy(config)

            script_config['deceiver_model'] = deceiver_model
            script_config['deceiver_config'] = deceiver_config
            script_config['category'] = category
            del script_config['deceiver_models']
            del script_config['deceiver_configs']
            del script_config['categories']

            logging_file_name = script_config["experiment_name"] + "-" + script_config['category']+ "-" + script_config['deceiver_config'] + "-%j.log"
            script_config['logging_file'] = logging_file_name

            script = generate_script(script_config)

            script_file_name = "run-job-" + script_config['category'] + "-" + script_config['deceiver_config'] + ".sbatch"
            script_save_path = os.path.join(script_config['script_saving_path'], script_config["experiment_name"], script_file_name)
            if not os.path.exists(os.path.join(script_config['script_saving_path'], script_config["experiment_name"])):
                os.makedirs(os.path.join(script_config['script_saving_path'], script_config["experiment_name"]))
            with open(script_save_path, 'w') as file:
                file.write(script)


# Example usage
llama_deceiving_config = {
        'experiment_name': 'run-2023-12-08',
        'logging_path': '/data/julius_heitkoetter/correlated_llm_errors/deploy/logging',
        'script_saving_path': '/data/julius_heitkoetter/correlated_llm_errors/deploy/batch_scripts',
        'working_directory': '/data/julius_heitkoetter/correlated_llm_errors',
        'dataset': 'mmlu',
        'categories': ['business_ethics','high_school_government_and_politics', 'high_school_world_history', 'international_law', 'management', 'marketing'],
        'save_location': 'hf',
        'deceiver_models': ['LlamaLLM', 'LlamaLLM', 'LlamaLLM'],
        'deceiver_configs': ['llama_7b_base_config', 'llama_13b_base_config', 'llama_70b_base_config'],
        'supervisor_models': ['LlamaLLM', 'LlamaLLM', 'LlamaLLM', 'OpenAILLM'],
        'supervisor_configs': ['llama_7b_base_config', 'llama_13b_base_config', 'llama_70b_base_config', 'gpt_35_turbo_base_config']
    }


if __name__ == "__main__":

    generate_multiple_scripts(llama_deceiving_config)
    print("Done!")
