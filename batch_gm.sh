#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=___
#SBATCH --qos=medium
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --output=output_files_log/fine_grained_log_Q%j.txt       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish 
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail  
#SBATCH --mem-per-cpu=3000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cudnn/7-cuda-10.0
# export CUDA_LAUNCH_BLOCKING=1

#source venv/bin/activate
# cd Glow_pyTorch/glow/
# python3 fine_grained.py

# python main_pl.py 

python main_fg_2.py -a resnet50  