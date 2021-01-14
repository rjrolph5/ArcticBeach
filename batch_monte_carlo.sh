#!/bin/bash

# water_level_solver.sbatch #
#SBATCH -D /permarisk/output/becca_erosion_model/ArcticBeach/ # Working directory that contains the water_level_solver function.
#SBATCH -J becca_monte_carlo_retreat_rates # A single job name for the array
#SBATCH -p PermaRisk # best partition for single core small jobs 
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH --nice=1000
#SBATCH --time="0-04" # days-HH
#SBATCH --output=/home/rrolph/erosion_model_batch_outfiles/out_batch/out_%a.txt
#SBATCH --error=/home/rrolph/erosion_model_batch_outfiles/err_batch/err_%a.txt
#SBATCH --array=1-500 # this is the number of times this batch script will be run. values or range used for testing was 1-10
# #SBATCH --mail-type=FAIL
# #SBATCH --mail-user=rebecca.rolph@awi.de

echo ${SLURMD_NODENAME}
echo ${SLURM_ARRAY_TASK_ID}

## Find the index number (slurm_array_task_id) of slurm and pass it to the python script to keep track of 
## the Monte Carlo iteration. Use sys to pass the index number to Python script.

$SLURM_ARRAY_TASK_ID

python3 water_level_solver_for_batch.py ${SLURM_ARRAY_TASK_ID}




