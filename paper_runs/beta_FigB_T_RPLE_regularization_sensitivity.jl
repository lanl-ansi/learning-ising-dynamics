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
FLAG_regular_random_gm = true
FLAG_lattice_gm = false
sampling_regime = T_regime()
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
    file_M_opt_gm = "M_opt_RR_FigB_T_RPLE_reg_sensitivity_beta_1_2.txt"
end

# Simple test
N = 16
d = 3
α = 0.4
β = 1.2

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
M_guess = 1000

# Different values of regularization coefficient to be tested
c_array = vcat([0.02,0.04],[0.05 + 0.05*i for i=0:9])
M_opt = Array{Int64,1}(undef,length(c_array))

# To prevent rewriting
adj_matrix_orig = deepcopy(adj_matrix)

@printf("Starting for Fig B, RPLE"); flush(stdout)
for i = 1:length(c_array)
    c = copy(c_array[i])
    @printf("c=%f \n", c); flush(stdout)
    @printf("beta=%f \n", β); flush(stdout)
    @printf("M_guess=%d \n", M_guess); flush(stdout)

    # Change the
    adj_matrix = replace_beta_gm(adj_matrix_orig, α, β)

    # Get the optimum number of samples
    M_opt[i] = get_M_opt_glauber_dynamics_regularization(adj_matrix, RPLE(c, true), learning_algo, sampling_regime, τ, L, M_guess, M_factor)

    # Update the guess of M_opt
    @printf("beta=%f, M_opt=%d \n", β, M_opt[i]); flush(stdout)
end

writedlm(file_M_opt_gm,M_opt)
