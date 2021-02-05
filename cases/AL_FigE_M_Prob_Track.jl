# Script to run a test case for understanding how the probability of success scales with active learning #

# Add paths
loc_gm_package = "../src/"
if !(loc_gm_package in LOAD_PATH)
    push!(LOAD_PATH, "../src/")
end

# Required packages
using GraphicalModelLearning
using LightGraphs
using Cairo
using Junet

# Function definitions
include("beta_scaling.jl")

# parameters
FLAG_create_struct_gm = true
FLAG_regular_random_gm = false
FLAG_lattice_gm = true
FLAG_weak_impurity = false
sampling_regime = M_regime()
learning_algo = CD(1e-12)

if FLAG_lattice_gm
    # text file to save adjacency matrix in
    file_adj_matrix_gm = "adj_matrix_ferro_lattice_gm_A_ME.txt"
    # name of picture to save graphical model in
    file_plot_gm = "ferro_lattice_gm_A_ME.eps"
    # File to save final results
    file_M_opt_gm = "M_opt_Ferro_Lattice_A_ME.txt"
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
d = 3
α = 0.4
β_array = [0.4+0.1*i for i=0:46]

# Create and plot the initial graphical graphical model
β = copy(β_array[4])
if FLAG_create_struct_gm
    if FLAG_regular_random_gm
        adj_matrix, struct_adj_matrix = spin_glass_random_regular_model(N,d,α,β)
        gm = Junet.Graph(struct_adj_matrix,directed=false)

        h = plot(gm, size=(300,300), layout=layout_fruchterman_reingold(gm), node_color="white", node_border_color="black",
            node_border_width=2, edge_color=[edge_color_gm(i, adj_matrix, α, β) for i = 1:edgecount(gm)],
            edge_width=2, format=:eps, file=file_plot_gm)
    elseif FLAG_lattice_gm
        adj_matrix, struct_adj_matrix = ferro_lattice(N,α,β,FLAG_weak_impurity)
        m = n = Int(sqrt(N))
        draw_periodic_lattice(adj_matrix,m,n,α,β,file_plot_gm)
        open(file_adj_matrix_gm, "w") do io
            writedlm(io, adj_matrix)
        end
    end
else
    adj_matrix = readdlm(file_adj_matrix_gm)
end

## Start the complexity studies
τ = α/2
L = 45
M_guess = 1000
M_factor = 0.05

M_opt = Array{Int64,1}(length(β_array))

if FLAG_lattice_gm
    for i = 1:length(β_array)
        # define β
        β = copy(β_array[i])
        println(β)

        # Create the adjacency matrix
        adj_matrix, struct_adj_matrix = ferro_lattice(N,α,β,FLAG_weak_impurity)

        # Get the optimum number of samples
        M_opt[i] = get_M_opt_glauber_dynamics(adj_matrix, RISE(), learning_algo, sampling_regime, τ, L, M_guess, M_factor)

        M_guess = Int(floor(0.8*copy(M_opt[i])))
    end
end

writedlm(file_M_opt_gm,M_opt)
