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
python bin/dataset_pipeline_deceiver_only.py mmlu high_school_psychology hf LlamaLLM llama_70b_base_config --num_samples 60
