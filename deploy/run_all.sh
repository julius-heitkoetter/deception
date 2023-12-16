##Runs all the slurm scripts in the folder it is run in

for sbatch_file in *.sbatch; do
    sbatch "$sbatch_file"
done