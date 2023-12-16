#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=36:00:00 
#SBATCH --partition=single
#SBATCH --gres=gpu:1
#SBATCH --job-name=CorrelatedErrorsQAEVEPipeline
#SBATCH --output=/data/julius_heitkoetter/correlated_llm_errors/deploy/logging/pipelinejob%j.log

echo Starting Setup
source ~/.bashrc
cd /data/julius_heitkoetter/correlated_llm_errors
conda activate correlated_errors
source install.sh
source setup.sh
echo Setup Complete

echo Starting Pipeline Run
python bin/dataset_pipeline_supervisors_only.py data/keep/mmlu_international-law_qae_Llama-2-13b-chat-hf_anyModel_2023-12-05-04-43-31-089950.json data/keep/mmlu_international-law_qae_Llama-2-13b-chat-hf_anyModel_2023-12-05-04-43-31-874168.json local OpenAILLM gpt_4_base_config
