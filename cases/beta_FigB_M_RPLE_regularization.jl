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
FLAG_create_struct_gm = false
FLAG_regular_random_gm = true
FLAG_lattice_gm = false
sampling_regime = M_regime()
FLAG_weak_impurity = false
learning_algo = NLP()

if FLAG_lattice_gm
    # text file to save adjacency matrix in
    file_adj_matrix_gm = "adj_matrix_ferro_lattice_gm_A_T.txt"
    # name of picture to save graphical model in
    file_plot_gm = "ferro_lattice_gm_A_T.eps"
    # File to save final results
    file_M_opt_gm = "M_opt_T_Regime_Ferro_Lattice_A.txt"
elseif FLAG_regular_random_gm
    # text file to save adjacency matrix in
    #file_adj_matrix_gm = "adj_matrix_ferro_random_regular_gm_B_M.txt"
    file_adj_matrix_gm = "spin_glass_random_regular_adj_matrix.txt"
    # name of picture to save graphical model in
    file_plot_gm = "ferro_random_regular_gm_B_M.eps"
    # File to save final results
    file_M_opt_gm = "M_opt_Regular_Random_B_M_RPLE_regularization.txt"
end

# Simple test
N = 16
d = 3
α = 0.4

β_array = [0.5 + 0.1*i for i=0:21]

# Create and plot the initial graphical graphical model
# Create and plot the initial graphical graphical model
β = copy(β_array[4])

if FLAG_create_struct_gm
    adj_matrix, struct_adj_matrix = spin_glass_random_regular_model(N,d,α,β)
else
    adj_matrix = readdlm(file_adj_matrix_gm)
    struct_adj_matrix = sign.(abs.(adj_matrix))
end
# As we are doing figure B
U_temp_adj_matrix = sparse(triu(struct_adj_matrix))

# adj_matrix is a sparse matrix
R, C, V = findnz(U_temp_adj_matrix)
V = β*V

# one random index to change to -α
ind_α = 8
if FLAG_weak_impurity
    V[ind_α] = -α
else
    V[ind_α] = α
end

U_adj_matrix = sparse(R, C, V, N, N)
adj_matrix = U_adj_matrix + U_adj_matrix'
adj_matrix = Array(adj_matrix)

## Start the complexity studies
τ = α/2
L = 45
M_factor = 0.05

M_opt = Array{Int64,1}(undef,length(β_array))

# To prevent rewriting
adj_matrix_orig = deepcopy(adj_matrix)

let
    M_guess = 7000
    for i = 1:length(β_array)
        # define β
        β = copy(β_array[i])
        println(β)
        println(M_guess)

        # Create the adjacency matrix
        adj_matrix = replace_beta_gm(adj_matrix_orig, α, β)

        # Get the optimum number of samples
        M_opt[i] = get_M_opt_glauber_dynamics_regularization(adj_matrix, RPLE(), learning_algo, sampling_regime, τ, L, M_guess, M_factor)

        # Update the guess of M_opt
        @printf("beta=%f, M_opt=%d \n", β, M_opt[i])
        M_guess = Int(floor(0.8*copy(M_opt[i])))
    end
end

writedlm(file_M_opt_gm,M_opt)
