# Script to run a test case for understanding how M scales with β #

# Add paths
loc_gm_package = "../src/"
if !(loc_gm_package in LOAD_PATH)
    push!(LOAD_PATH, "../src/")
end

# Required packages
using LinearAlgebra

using GML_Glauber_Dynamics
using LightGraphs

# Function definitions
using Printf, SparseArrays
include("beta_scaling.jl")

using DelimitedFiles

# parameters
FLAG_create_struct_gm = false
FLAG_regular_random_gm = false
FLAG_lattice_gm = true
sampling_regime = T_regime()
learning_algo = NLP()

if FLAG_lattice_gm
    # text file to save adjacency matrix in
    file_adj_matrix_gm = "adj_matrix_ferro_lattice_gm_C_T.txt"
    # name of picture to save graphical model in
    file_plot_gm = "ferro_lattice_gm_C_T.eps"
    # File to save final results
    file_M_opt_gm = "M_opt_Ferro_Lattice_C_T_RPLE.txt"
elseif FLAG_regular_random_gm
    # text file to save adjacency matrix in
    #file_adj_matrix_gm = "adj_matrix_ferro_random_regular_gm_B_M.txt"
    file_adj_matrix_gm = "spin_glass_random_regular_adj_matrix.txt"
    # name of picture to save graphical model in
    file_plot_gm = "ferro_random_regular_gm_F_T.eps"
    # File to save final results
    file_M_opt_gm = "M_opt_M_Regime_Regular_Random_C_T.txt"
end

# Simple test
N = 16
d = 4
α = 0.4
c = 0.1
β_array = [0.8 + i*0.1 for i=0:12]

# Create and plot the initial graphical graphical model
adj_matrix = readdlm(file_adj_matrix_gm)
struct_adj_matrix = sign.(abs.(adj_matrix))

## Start the complexity studies
τ = α/2
L = 45
M_factor = 0.05

M_opt = Array{Int64,1}(undef,length(β_array))

# To prevent rewriting
adj_matrix_orig = deepcopy(adj_matrix)

let
    M_guess = 1000
    for i = 1:length(β_array)
        # define β
        β = copy(β_array[i])
        @printf("beta=%f \n", β); flush(stdout)
        @printf("M_guess=%d \n", M_guess); flush(stdout)

        # Create the adjacency matrix
        adj_matrix = replace_beta_gm(adj_matrix_orig, α, β)

        # Get the optimum number of samples
        M_opt[i] = get_M_opt_glauber_dynamics_regularization(adj_matrix, RPLE(c, true), learning_algo, sampling_regime, τ, L, M_guess, M_factor)

        # Update the guess of M_opt
        @printf("beta=%f, M_opt=%d \n", β, M_opt[i])
        M_guess = Int(floor(1.3*copy(M_opt[i])))
    end
end

writedlm(file_M_opt_gm,M_opt)
