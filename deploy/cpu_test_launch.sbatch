#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --partition=single 
#SBATCH --job-name=CorrelatedErrorsTestJob 
#SBATCH --output=testjob%j.log
source ~/.bashrc

cd ~/correlated_llm_errors

conda activate correlated_errors
echo activated conda environment

source install.sh
echo setup.sh created

source setup.sh
echo setup.sh run

echo starting test script
python ./test/dataloader_test.py
