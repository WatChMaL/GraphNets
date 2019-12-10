#!/bin/bash
#SBATCH --job-name=validate_graphnet
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --cpus-per-task=4         # NUmber of CPU cores/thread
#SBATCH --mem=16000M              # Memory (per node)
#SBATCH --time=2-00:00            # time (DD-HH:MM)

#SBATCH --account=def-pdeperio

#SBATCH --output=log-validate-%u-%j.txt
#SBATCH --error=error-validate-%u-%j.txt

## Make sure the following environmental variables are set
# SINGULARITY_IMAGE
# DATA_DIR
# PROJECT_DIR

## Example setup
# export SINGULARITY_IMAGE=/project/rpp-tanaka-ab/wollip/graph_ml_pytorch_geometric_dev.sif
# export DATA_DIR=/project/rpp-tanaka-ab/machine_learning/IWCDmPMT_4pi_full_tank/h5/graphnet
### This folder will copied to local storage (<800GB) 
###     then mounted to /fast_scratch inside of the singularity image
# export PROJECT_DIR=/project/rpp-tanaka-ab/wollip/GraphNet
### This will be mounted to /project_dir inside of the singularity image

module load singularity/3.2

date
echo "Copying data from $DATA_DIR to $SLURM_TMPDIR"
rsync -r $DATA_DIR $SLURM_TMPDIR

date
echo "Start running job"

singularity exec --nv -B $SLURM_TMPDIR:/fast_scratch -B $PROJECT_DIR:/project_dir $SINGULARITY_IMAGE python /project_dir/GraphNets/validate_custom.py

date
echo "Job Done"
