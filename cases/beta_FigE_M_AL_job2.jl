# Script to run a test case for understanding how M scales with β #

# Add paths
loc_gm_package = "../src/"
if !(loc_gm_package in LOAD_PATH)
    push!(LOAD_PATH, "../src/")
end

# Required packages
using LinearAlgebra
using Convex, SCS, Mosek, MosekTools

using GraphicalModelLearning
using LightGraphs

# Function definitions
using Printf, SparseArrays
include("beta_scaling.jl")
include("active_learning.jl")

using DelimitedFiles

# parameters
FLAG_create_struct_gm = true
FLAG_regular_random_gm = false
FLAG_lattice_gm = true
FLAG_weak_impurity = true
sampling_regime = M_regime()
learning_algo = NLP()

if FLAG_lattice_gm
    # text file to save adjacency matrix in
    file_adj_matrix_gm = "adj_matrix_ferro_lattice_gm_E_MEE2.txt"
    # name of picture to save graphical model in
    file_plot_gm = "ferro_lattice_gm_E_MEE2.eps"
    # File to save final results
    file_M_opt_gm = "M_opt_Ferro_Lattice_E_MEE2.txt"
elseif FLAG_regular_random_gm
    # text file to save adjacency matrix in
    file_adj_matrix_gm = "adj_matrix_spin_glass_gm.txt"
    # name of picture to save graphical model in
    file_plot_gm = "spin_glass_gm3.eps"
    # File to save final results
    file_M_opt_gm = "M_opt_M_Regime_Regular_Random_B.txt"
end

# Simple test
N = 16
d = 4
α = 0.4
β_array = [1.8,3.2]

# Create and plot the initial graphical graphical model
β = copy(β_array[1])
adj_matrix, struct_adj_matrix = ferro_lattice(N,α,β,FLAG_weak_impurity)
m = n = Int(sqrt(N))
open(file_adj_matrix_gm, "w") do io
    writedlm(io, adj_matrix)
end;

## Start the complexity studies
τ = α/2
L = 45
M_g = 25000
M_factor = 0.05

M_opt = Array{Int64,1}(undef,length(β_array))

for i = 1:length(β_array)
    # define β
    β = copy(β_array[i])
    println(β)

    if i==1
        M_g = 25000
    end
    println(M_g)

    # Create the adjacency matrix
    adj_matrix, struct_adj_matrix = ferro_lattice(N,α,β,FLAG_weak_impurity)

    # Get the optimum number of samples
    M_opt[i] = get_M_opt_glauber_dynamics_AL(adj_matrix, τ, L, M_g, M_factor)

    # Update the guess of M_opt
    @printf("beta=%f, M_opt=%d \n", β, M_opt[i])

    M_g = Int(floor(1.8*M_opt[i]))
    println(M_g)
end

writedlm(file_M_opt_gm,M_opt)
