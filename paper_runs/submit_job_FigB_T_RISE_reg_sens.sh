#!/bin/bash

# Slurm sbatch options
#SBATCH -o glauber_dynamics_reg.log-%j
#SBATCH -n 1

# Initialize the module command first
source /etc/profile

# Load the module
module load julia/1.5.2 

# Call your script as you would from the command line with the environment mentioned
julia --project=~/Research/Graphical_Model_Learning/Code/GML_Glauber_Dynamics/ beta_FigB_T_RISE_regularization_sensitivity.jl



