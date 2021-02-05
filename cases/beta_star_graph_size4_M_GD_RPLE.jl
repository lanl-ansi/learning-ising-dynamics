# Script to run a test case for understanding how M scales with β #

# Add paths
loc_gm_package = "../src/"
if !(loc_gm_package in LOAD_PATH)
    push!(LOAD_PATH, "../src/")
end

# Required packages
using LinearAlgebra

using GraphicalModelLearning
using LightGraphs

# Function definitions
using Printf, SparseArrays
include("beta_scaling.jl")

using DelimitedFiles

# parameters
FLAG_create_struct_gm = true
FLAG_regular_random_gm = true
FLAG_lattice_gm = false
sampling_regime = M_regime()
learning_algo = NLP()

@printf("Creating a star graph of size 4!")
file_M_opt_gm = "M_opt_stargraph_s4_M_RPLE.txt"

# Simple test
N = 3
d = 2
α = 0.4
β_array = [2.0 + 0.1*i for i=8:12]

# Create and plot the initial graphical graphical model
β = copy(β_array[4])
adj_matrix = [0 β β α; β 0 0 0; β 0 0 0; α 0 0 0]
struct_adj_matrix = sign.(abs.(adj_matrix))

## Start the complexity studies
τ = α/2
L = 45
M_factor = 0.05
M_opt = Array{Int64,1}(undef,length(β_array))

# To prevent rewriting
adj_matrix_orig = deepcopy(adj_matrix)

let
    M_guess = 91000
    for i = 1:length(β_array)
        # define β
        β = copy(β_array[i])
        println(β)
        println(M_guess)

        # Create the adjacency matrix
        adj_matrix = replace_beta_gm(adj_matrix_orig, α, β)

        # Get the optimum number of samples
        M_opt[i] = get_M_opt_glauber_dynamics(adj_matrix, RPLE(0.0, true), learning_algo, sampling_regime, τ, L, M_guess, M_factor)

        # Update the guess of M_opt
        @printf("beta=%f, M_opt=%d \n", β, M_opt[i])
        M_guess = Int(floor(0.8*copy(M_opt[i])))
    end
end

writedlm(file_M_opt_gm,M_opt)
