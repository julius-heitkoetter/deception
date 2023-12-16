#!/bin/bash
#---------------------------------------------------------------------------------------------------
# Create slurm script which runs a single pass of the pipeline
#---------------------------------------------------------------------------------------------------

# Setup folder for logging 
mkdir -p $BASE/deploy/logging 

# Ask for input parameters
echo "Enter dataset name (mmlu or ethics):"
read dataset_name

echo "Enter category:"
read category

echo "Enter save location (local or hf):"
read save_location

echo "Enter deceiver model name:"
read deceiver_model_name

echo "Enter deceiver config name:"
read deceiver_config_name

echo "Enter supervisor model name:"
read supervisor_model_name

echo "Enter supervisor config name:"
read supervisor_config_name

echo "Enter number of samples (optional, press Enter to skip):"
read num_samples

# Create the sbatch script
sbatch_script="$BASE/deploy/run_pipeline_job.sbatch"

echo "#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --time=24:00:00 
#SBATCH --partition=single
#SBATCH --gres=gpu:1
#SBATCH --job-name=CorrelatedErrorsQAEVEPipeline
#SBATCH --output=$BASE/deploy/logging/pipelinejob%j.log

echo Starting Setup
source ~/.bashrc
cd $BASE
conda activate correlated_errors 
source install.sh 
source setup.sh
echo Setup Complete

echo Starting Pipeline Run" > $sbatch_script

# Check if num_samples is provided
if [ -z "$num_samples" ]
then
    echo "python bin/dataset_pipeline.py $dataset_name $category $save_location $deceiver_model_name $deceiver_config_name $supervisor_model_name $supervisor_config_name" >> $sbatch_script
else
    echo "python bin/dataset_pipeline.py $dataset_name $category $save_location $deceiver_model_name $deceiver_config_name $supervisor_model_name $supervisor_config_name --num_samples $num_samples" >> $sbatch_script
fi

echo "sbatch script created: $sbatch_script"
