## Defines functions to be used for beta scaling studies

# create a ferromagnet on a d random regular graph with or without weak impurity
function ferro_random_regular_model(N::Integer,d::Integer,α::Real,β::Real,FLAG_weak_impurity=false)
    #=
    # choose with equal probability those to be changed to -β
    # should select random numbers uniformly without replacement
    # also need to make sure resulting matrix is symmetric
    =#

    rand_reg_gm = random_regular_graph(N,d)
    temp_adj_matrix = adjacency_matrix(rand_reg_gm)

    U_temp_adj_matrix = triu(temp_adj_matrix)

    # adj_matrix is a sparse matrix
    R, C, V = findnz(U_temp_adj_matrix)
    V = β*V

    # one random index to change to -α
    ind_α = 4
    if FLAG_weak_impurity
        V[ind_α] = -α
    else
        V[ind_α] = α
    end

    U_adj_matrix = sparse(R, C, V, N, N)
    adj_matrix = U_adj_matrix + U_adj_matrix'
    adj_matrix = Array(adj_matrix)

    # Needed for plotting
    struct_adj_matrix = Array(temp_adj_matrix)

    return adj_matrix, struct_adj_matrix
end

# create a spin glass model on a d random regular graph
function spin_glass_random_regular_model(N::Integer,d::Integer,α::Real,β::Real)
    #=
    # choose with equal probability those to be changed to -β
    # should select random numbers uniformly without replacement
    # also need to make sure resulting matrix is symmetric
    =#

    rand_reg_gm = random_regular_graph(N,d)
    temp_adj_matrix = adjacency_matrix(rand_reg_gm)

    U_temp_adj_matrix = triu(temp_adj_matrix)

    # adj_matrix is a sparse matrix
    R, C, V = findnz(U_temp_adj_matrix)
    V = β*V

    # indices to be changed to negative β (randperm or randsubseq)
    # Ref: https://discourse.julialang.org/t/sampling-without-replacement/1073
    #ind_neg_β = randsubseq(1:length(V),0.5)
    ind_neg_β = randperm(length(V))[1:Int(N/2)]

    # Also find two random indices to change to +/- α
    ind_α = randperm(length(V))[1:2]

    V[ind_neg_β] = -β
    V[ind_α[1]] = α
    V[ind_α[2]] = -α

    U_adj_matrix = sparse(R, C, V, N, N)
    adj_matrix = U_adj_matrix + U_adj_matrix'
    adj_matrix = Array(adj_matrix)

    # Needed for plotting
    struct_adj_matrix = Array(temp_adj_matrix)

    return adj_matrix, struct_adj_matrix
end

# create a spin glass model on a double periodic lattice
function spin_glass_lattice(N::Integer,α::Real,β::Real)
    #=
    # choose with equal probability those to be changed to -β
    # should select random numbers uniformly without replacement
    # also need to make sure resulting matrix is symmetric
    =#
    temp_adj_matrix = double_periodic_lattice(Int(sqrt(N)),Int(sqrt(N)))
    temp_adj_matrix = sparse(temp_adj_matrix)

    U_temp_adj_matrix = triu(temp_adj_matrix)

    # adj_matrix is a sparse matrix
    R, C, V = findnz(U_temp_adj_matrix)
    V = β*V

    # indices to be changed to negative β (randperm or randsubseq)
    # Ref: https://discourse.julialang.org/t/sampling-without-replacement/1073
    #ind_neg_β = randsubseq(1:length(V),0.5)
    ind_neg_β = randperm(length(V))[1:Int(N/2)]

    # Also find two random indices to change to +/- α
    ind_α = randperm(length(V))[1:2]
    # HACK FOR NOW
    ind_α[1] = 4
    ind_α[2] = 28

    V[ind_neg_β] = -β
    V[ind_α[1]] = α
    V[ind_α[2]] = -α

    U_adj_matrix = sparse(R, C, V, N, N)
    adj_matrix = U_adj_matrix + U_adj_matrix'
    adj_matrix = Array(adj_matrix)

    # Needed for plotting
    struct_adj_matrix = Array(temp_adj_matrix)

    return adj_matrix, struct_adj_matrix
end

# replace values of beta in an existing spin glasss graphical model
function replace_beta_gm(adj_matrix::Array, α::Real, β::Real)
    mod_adj_matrix = deepcopy(adj_matrix)

    # Replace values greater than α
    mod_adj_matrix[adj_matrix.>α] .= β

    # Replace values smaller than -α
    mod_adj_matrix[adj_matrix.<-α] .= -β

    return mod_adj_matrix
end

# create a double periodic lattice of size (m,n)
function double_periodic_lattice(m::Integer,n::Integer)
    n_spins = m*n
    adj_matrix = zeros(n_spins,n_spins)

    for i = 1:(m*n)
        # top
        adj_matrix[i,( ((i-1) % m) == 0 ? (i+m-1) : i-1 )] = 1

        # bottom
        adj_matrix[i,( ((i+1) % m) == 1 ? (i+1-m) : i+1 )] = 1

        # left
        adj_matrix[i,( (i+m) > n_spins ? (i+m) % n_spins : i+m )] = 1

        # right
        adj_matrix[i,( (i-m) <= 0 ? (n_spins-m+i) : i-m )] = 1
    end

    return adj_matrix
end

# create a periodic ferromagnetic lattice with or without weak impurity
function ferro_lattice(N::Integer,α::Real,β::Real,FLAG_weak_impurity=false)
    temp_adj_matrix = double_periodic_lattice(Int(sqrt(N)),Int(sqrt(N)))
    temp_adj_matrix = sparse(temp_adj_matrix)

    U_temp_adj_matrix = triu(temp_adj_matrix)

    # adj_matrix is a sparse matrix
    R, C, V = findnz(U_temp_adj_matrix)
    V = β*V

    # one random index to change to -α
    #ind_α = rand(1:length(V))
    ind_α = 4
    if FLAG_weak_impurity
        V[ind_α] = -α
    else
        V[ind_α] = α
    end

    U_adj_matrix = sparse(R, C, V, N, N)
    adj_matrix = U_adj_matrix + U_adj_matrix'
    adj_matrix = Array(adj_matrix)

    # Needed for plotting
    struct_adj_matrix = Array(temp_adj_matrix)

    return adj_matrix, struct_adj_matrix
end

# create a non periodic lattice of size (m,n) for plotting purposes
function non_periodic_lattice(adj_matrix_gm::Array, m::Integer,n::Integer)
    # Map of the nodes in the plotted gm
    node_map_plot = Array{Int64,2}(undef,m,n)
    n_spins = m*n
    adj_matrix_plot = zeros(n_spins,n_spins)
    node_map_plot[1:n_spins] = 1:n_spins

    # Map of the indices in the plotted map to the original map
    values_map_plot = Array{Int64,2}(undef,m,n)
    values_map_plot[2:m-1,2:n-1] = 1:((m-2)*(n-2))

    # Specify edges between nodes at the boundary to be removed
    boundary_nodes = vcat(node_map_plot[:,1],node_map_plot[:,n],node_map_plot[1,2:n-1],node_map_plot[m,2:n-1])
    corner_nodes = [1,m,m*(n-1)+1,m*n]

    # Apply periodicity
    values_map_plot[2:(m-1),1] = copy(values_map_plot[2:(m-1),(n-1)])
    values_map_plot[2:(m-1),n] = copy(values_map_plot[2:(m-1),2])
    values_map_plot[1,2:(n-1)] = copy(values_map_plot[(m-1),2:(n-1)])
    values_map_plot[m,2:(n-1)] = copy(values_map_plot[2,2:(n-1)])
    values_map_plot[corner_nodes] = zeros(length(corner_nodes))

    # Create edges
    for i = 1:(m*n)
        if !(i in corner_nodes)
            # top
            if ((i-1) % m) != 0 && !((i-1) in corner_nodes)
                adj_matrix_plot[i,i-1] = adj_matrix[values_map_plot[i],values_map_plot[i-1]]
            end
            # bottom
            if ((i+1) % m) != 1 && !((i+1) in corner_nodes)
                adj_matrix_plot[i,i+1] = adj_matrix[values_map_plot[i],values_map_plot[i+1]]
            end
            # right
            if (i+m) <= n_spins && !((i+m) in corner_nodes)
                adj_matrix_plot[i,i+m] = adj_matrix[values_map_plot[i],values_map_plot[i+m]]
            end
            # left
            if (i-m) > 0 && !((i-m) in corner_nodes)
                adj_matrix_plot[i,i-m] = adj_matrix[values_map_plot[i],values_map_plot[i-m]]
            end
        end
    end
    for i in boundary_nodes
        for j in boundary_nodes
            adj_matrix_plot[i,j] = 0
        end
    end

    struct_adj_matrix_plot = sign.(abs.(adj_matrix_plot))

    return adj_matrix_plot, struct_adj_matrix_plot
end

# Function to draw a periodic lattice using Junet (no longer works in Julia 1.0)
function draw_periodic_lattice(adj_matrix::Array, m::Integer, n::Integer, α::Real, β::Real, gm_file::String)
    # Get the graph that Junet can plot
    adj_matrix_junet, struct_adj_matrix_junet = non_periodic_lattice(adj_matrix,m+2,n+2)

    # Create Junet Graph
    gm = Junet.Graph(struct_adj_matrix_junet,directed=false)

    # Layout
    N = nodecount(gm)
    x = Array{Float64,1}(N)
    y = Array{Float64,1}(N)

    node_map_gm = zeros(m+2,n+2)
    node_map_gm[1:N] = 1:N

    for i=1:N
        y[i], x[i] = ind2sub(node_map_gm,find(isequal(i),node_map_gm)[1])
        if i in node_map_gm[:,1]
            x[i] += 0.3
        elseif i in node_map_gm[:,n+2]
            x[i] += -0.3
        elseif i in node_map_gm[1,:]
            y[i] += +0.3
        elseif i in node_map_gm[m+2,:]
            y[i] += -0.3
        end
    end

    # Decide node border colors
    node_border_colors_gm = Array{String,1}(N)
    for i=1:N
        if (i % (m+2) == 1) || (i % (m+2) == 0) || (i <= m+2) || (i >= (m+2)*(n+1))
            node_border_colors_gm[i] = "white"
        else
            node_border_colors_gm[i] = "black"
        end
    end

    h = Junet.plot(gm, size=(300,300), layout=(x,y), node_color="white", node_border_color=node_border_colors_gm,
        node_border_width=2, edge_color=[edge_color_gm(i, adj_matrix_junet, α, β) for i = 1:edgecount(gm)],
        edge_width=3, format=:eps, file=gm_file)
end


# Function to draw a periodic lattice using GraphPlot.jl
function draw_periodic_lattice2(adj_matrix::Array, m::Integer, n::Integer, α::Real, β::Real)
    # Get the graph that Junet can plot
    adj_matrix_plot, struct_adj_matrix_plot = non_periodic_lattice(adj_matrix,m+2,n+2)

    # Layout
    N = (m+2)*(n+2)
    x = Array{Float64,1}(undef,N)
    y = Array{Float64,1}(undef,N)

    node_map_gm = zeros(m+2,n+2)
    node_map_gm[1:N] = 1:N

    for i=1:N
        y[i], x[i] = Tuple(findfirst(isequal(i),node_map_gm))
        if i in node_map_gm[:,1]
            x[i] += 0.3
        elseif i in node_map_gm[:,n+2]
            x[i] += -0.3
        elseif i in node_map_gm[1,:]
            y[i] += +0.3
        elseif i in node_map_gm[m+2,:]
            y[i] += -0.3
        end
    end

    # Decide node border colors
    node_border_colors_gm = Array{String,1}(undef,N)
    for i=1:N
        if (i % (m+2) == 1) || (i % (m+2) == 0) || (i <= m+2) || (i >= (m+2)*(n+1))
            node_border_colors_gm[i] = "white"
        else
            node_border_colors_gm[i] = "black"
        end
    end

    gm = SimpleGraph(struct_adj_matrix_plot)
    gplot(gm, x, y)
end


# decide the edge color to be used when plotting a graphical model
function edge_color_gm(ind_edge::Integer, adj_matrix::Array, α::Real, β::Real)
    A = sparse(adj_matrix)
    R, C, V = findnz(A)
    v = V[ind_edge]

    if v ≈ α
        edge_color="orange"
    elseif v ≈ β
        edge_color="red"
    elseif v ≈ -α
        edge_color="cyan"
    elseif v ≈ -β
        edge_color="blue"
    end

    return edge_color
end

# Assert if the correct gm has been obtained
function assert_correct_gm_old(learned_gm::Array, true_gm::Array, τ::Real)
    learned_struct_gm = zeros(size(learned_gm))

    for i = 1:size(learned_gm,1)
        for j = 1:size(learned_gm,2)
            if abs(learned_gm[i,j]) <= τ
                learned_struct_gm[i,j] = 0
            else
                learned_struct_gm[i,j] = sign(learned_gm[i,j])
            end
        end
    end

    true_struct_gm = sign.(true_gm)

    return all(learned_struct_gm .≈ true_struct_gm), learned_struct_gm, true_struct_gm
    #return all(learned_struct_gm .≈ true_struct_gm)
end

# Assert if the correct gm has been obtained
function assert_correct_gm(learned_gm::Array, true_gm::Array, τ::Real)
    learned_struct_gm = Array{Int8,2}(undef,size(learned_gm))
    true_struct_gm = Array{Int8,2}(undef,size(true_gm))

    for i = 1:size(learned_gm,1)
        for j = 1:size(learned_gm,2)
            if abs(learned_gm[i,j]) < τ
                learned_struct_gm[i,j] = 0
            else
                learned_struct_gm[i,j] = sign(learned_gm[i,j])
            end
        end
    end

    true_struct_gm = sign.(true_gm)

    #return all(learned_struct_gm .== true_struct_gm), learned_struct_gm, true_struct_gm
    return all(learned_struct_gm .== true_struct_gm)
end

# Assert if the correct gm has been obtained
function assert_correct_gm2(learned_gm::Array, true_gm::Array, τ::Real)
    learned_struct_gm = Array{Int8,2}(size(learned_gm))
    true_struct_gm = Array{Int8,2}(size(true_gm))

    for i = 1:size(learned_gm,1)
        for j = 1:size(learned_gm,2)
            if abs(learned_gm[i,j]) < τ
                learned_struct_gm[i,j] = 0
            else
                learned_struct_gm[i,j] = sign(learned_gm[i,j])
            end
        end
    end

    true_struct_gm = sign.(true_gm)

    return all(learned_struct_gm .== true_struct_gm), learned_struct_gm, true_struct_gm
end

# get the optimal number of samples for a given graphical model -- Glauber Dynamics
function get_M_opt(true_adj_matrix::Array, learning_method::GMLFormulation, learning_algo::GMLMethod, τ=0.2, L_success=45, M_guess=1000, M_factor=0.1)
    # Graphical model
    true_gm = FactorGraph(true_adj_matrix)

    # Initialize
    N_trials = 0    # Number of attempts
    L_trials = 0    # Number of successful trials so far
    M = copy(M_guess)   # Number of samples
    FLAG_correct_gm = false # FLAG if correct gm is recovered
    samples = sample(true_gm, M)

    while L_success > L_trials
        # Learn the graphical model
        learned_adj_matrix = learn(samples, learning_method, learning_algo)

        # Assert if correct GM
        FLAG_correct_gm = assert_correct_gm(learned_adj_matrix, true_adj_matrix, τ)

        if FLAG_correct_gm
            N_trials += 1
            L_trials += 1
            samples = sample(true_gm, M)
        else
            N_trials = 0
            L_trials = 0
            M = Int(floor(((1 + M_factor)*M)/100))*100
            samples = sample(true_gm, M)
        end
        @printf("FLAG=%d, M=%d, n_trials=%d, P=%d/%d\n", FLAG_correct_gm, M, N_trials, L_trials, L_success)
    end

    return M
end

# get the optimal number of samples for a given graphical model -- Glauber Dynamics
function get_M_opt_glauber_dynamics(true_adj_matrix::Array, learning_method::GMLFormulation, learning_algo::GMLMethod, sampling_regime::SamplingRegime, τ=0.2, L_success=45, M_guess=1000, M_factor=0.1)
    # Graphical model
    true_gm = FactorGraph(true_adj_matrix)

    # Initialize
    N_trials = 0    # Number of attempts
    L_trials = 0    # Number of successful trials so far
    M = copy(M_guess)   # Number of samples
    FLAG_correct_gm = false # FLAG if correct gm is recovered
    samples_T, samples_mixed = gibbs_sampling2(true_gm, M, sampling_regime)

    while L_success > L_trials
        # Learn the graphical model
        learned_adj_matrix = learn_glauber_dynamics(samples_T, learning_method, learning_algo)

        # Assert if correct GM
        FLAG_correct_gm = assert_correct_gm(learned_adj_matrix, true_adj_matrix, τ)

        if FLAG_correct_gm
            N_trials += 1
            L_trials += 1
            samples_T, samples_mixed = gibbs_sampling2(true_gm, M, sampling_regime)
        else
            N_trials = 0
            L_trials = 0
            M = Int(floor(((1 + M_factor)*M)/50))*50
            samples_T, samples_mixed = gibbs_sampling2(true_gm, M, sampling_regime)
        end
        @printf("FLAG=%d, M=%d, n_trials=%d, P=%d/%d\n", FLAG_correct_gm, M, N_trials, L_trials, L_success)
    end

    return M
end


# get the optimal number of samples for a given graphical model -- Glauber Dynamics
function get_M_opt_glauber_dynamics_regularization(true_adj_matrix::Array, learning_method::GMLFormulation, learning_algo::GMLMethod, sampling_regime::SamplingRegime, τ=0.2, L_success=45, M_guess=1000, M_factor=0.1)
    # Graphical model
    true_gm = FactorGraph(true_adj_matrix)

    # Initialize
    N_trials = 0    # Number of attempts
    L_trials = 0    # Number of successful trials so far
    M = copy(M_guess)   # Number of samples
    FLAG_correct_gm = false # FLAG if correct gm is recovered
    samples_T, samples_mixed = gibbs_sampling2(true_gm, M, sampling_regime)

    while L_success > L_trials
        try
            # Learn the graphical model
            learned_adj_matrix = GML_Glauber_Dynamics.learn_glauber_dynamics_regularization(samples_T, learning_method, learning_algo)

            # Assert if correct GM
            FLAG_correct_gm = assert_correct_gm(learned_adj_matrix, true_adj_matrix, τ)
        catch
            @info("Most probably LoadError:AssertionError in JuMP")
            FLAG_correct_gm = false
        end

        if FLAG_correct_gm
            N_trials += 1
            L_trials += 1
            samples_T, samples_mixed = gibbs_sampling2(true_gm, M, sampling_regime)
        else
            N_trials = 0
            L_trials = 0
            M = Int(floor(((1 + M_factor)*M)/50))*50
            samples_T, samples_mixed = gibbs_sampling2(true_gm, M, sampling_regime)
        end
        @printf("FLAG=%d, M=%d, n_trials=%d, P=%d/%d\n", FLAG_correct_gm, M, N_trials, L_trials, L_success)
        flush(stdout)
    end

    return M
end


# Function to carry out one active learning return
function active_learning_run(m0::Integer, mtot::Integer, mbatch::Integer, num_spins::Integer, true_adj_matrix::Array, learning_method::GMLFormulation, learning_algo::GMLMethod, τ=0.2, FLAG_verbose=true)
    #=
    Inputs:
    m0 -  size of the first batch
    mbatch - preferred size of samples in each batch
    mtot - total number of samples
    true_adj_matrix is basically my oracle here

    Assuming the sampling regime is M-regime and hence not mentioned here
    =#
    # Mon oracle
    true_gm = FactorGraph(true_adj_matrix)

    # Create alphabet
    config_number = 2^num_spins
    X_U = [config for config=0:(config_number -1)]
    p_U = (1/config_number)*ones(config_number)

    # Get the initial set of samples
    samples_M, samples_mixed = gibbs_sampling_query(true_gm, m0, X_U, p_U, M_regime())
    m_remaining = mtot-m0
    M = copy(m0)

    # learn the graphical model based on the samples
    learned_adj_matrix = learn_glauber_dynamics(samples_M, RISE(), NLP())

    if FLAG_verbose
        FLAG_correct_gm = assert_correct_gm(learned_adj_matrix, true_adj_matrix, τ)
        @printf("Active: FLAG=%d, M=%d \n", FLAG_correct_gm, M)
    end

    while m_remaining > 0
        # create the entropy distribution
        s_configs_learned = entropy_configs(learned_adj_matrix,X_U,n)
        q_s_learned = s_configs_learned/sum(s_configs_learned)

        # modify the distribution slightly by adding a small amount of uniform distribution
        lambda = 1 - 1/((M)^(1/6))
        q_s = lambda*q_s_learned + (1-lambda)*p_U

        # get a batch of samples and update remaining number of queries
        mtemp = min(m_remaining,mbatch)
        samples_M_temp, samples_mixed_temp = gibbs_sampling_query(true_gm, mtemp, X_U, q_s, M_regime(), false)

        m_remaining = m_remaining - mtemp
        M += mtemp

        # add to the existing set of samples
        samples_M = GraphicalModelLearning.add_histograms(samples_M,samples_M_temp)

        # learn the graphical model based on the samples
        learned_adj_matrix = learn_glauber_dynamics(samples_M, RISE(), NLP())

        if FLAG_verbose
            FLAG_correct_gm = assert_correct_gm(learned_adj_matrix, true_adj_matrix, τ)
            @printf("Active: FLAG=%d, M=%d \n", FLAG_correct_gm, M)
        end
    end

    return learned_adj_matrix
end


# get the optimal number of samples for a given graphical model -- Glauber Dynamics
function get_M_opt_glauber_dynamics_AL(true_adj_matrix::Array, τ=0.2, L_success=45, M_guess=1000, M_factor=0.1)
    # Graphical model
    true_gm = FactorGraph(true_adj_matrix)
    n = true_gm.varible_count

    # Initialize
    N_trials = 0    # Number of attempts
    L_trials = 0    # Number of successful trials so far
    M = copy(M_guess)   # Number of samples
    FLAG_correct_gm = false # FLAG if correct gm is recovered

    learned_adj_matrix = Array{Float64,2}(undef,size(true_adj_matrix))

    # Parameters of run
    #m0 = min(Int(floor(M/2)), 5000)
    m0 = Int(floor(M/3))
    mbatch = Int(floor((M-m0)/15))

    while L_success > L_trials
        # Carry out an independent active learning run
        learned_adj_matrix = active_learning_run(m0, M, mbatch, n, true_adj_matrix, RISE(), NLP(), τ, false)

        # Assert if correct GM
        FLAG_correct_gm = assert_correct_gm(learned_adj_matrix, true_adj_matrix, τ)

        if FLAG_correct_gm
            N_trials += 1
            L_trials += 1
        else
            N_trials = 0
            L_trials = 0
            M = Int(floor(((1 + M_factor)*M)/50))*50

            m0 = Int(floor(M/3))
            mbatch = Int(floor((M-m0)/15))
        end
        @printf("FLAG=%d, M=%d, n_trials=%d, P=%d/%d\n", FLAG_correct_gm, M, N_trials, L_trials, L_success)
    end

    return M
end


# Evaluate the gradient of the objective function afterwards
function get_gradient_value(samples::Array, learned_gm::Array, learning_method::GMLFormulation, learning_method_name::String)
    num_conf, num_spins, num_samples = data_info(samples)

    grad_value = 0
    grad_value_i_array = Array{Float64,1}(num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + num_spins + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf_i , i=1:num_spins]

        x = learned_gm[current_spin,:]

        if learning_method_name == "RISE"
            iso_obj_deriv(x,k) = (1/num_samples)*( sum(samples[:,1].*(exp.(-1*(nodal_stat*x))).*(-1*nodal_stat[:,k])) )
        elseif learning_method_name == "RPLE"
            iso_obj_deriv(x,k) = (1/num_samples)*( sum(samples[:,1].*(1 ./ (1 + exp.(2*(nodal_stat*x)))).*(-2*nodal_stat[:,k])) )
        end

        for ind_spin = 1:num_spins
            grad_value_i_array[ind_spin] = iso_obj_deriv(x,ind_spin)
        end
        grad_value += norm(grad_value_i_array,Inf)
    end
    return grad_value/num_spins
end

# Evaluate the gradient of the objective function afterwards
function get_gradient_value_glauber_dynamics(samples::Array, learned_gm::Array, learning_method::GMLFormulation, learning_method_name::String)
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    grad_value = 0
    grad_value_i_array = Array{Float64,1}(num_spins)

    for current_spin = 1:num_spins
        samples_i = samples[find(isequal(current_spin),samples[:,2]),:]
        num_conf_i = size(samples_i,1)
        num_samples_i = sum(samples_i[:,1])

        nodal_stat  = [ samples_i[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i[k, 2 + i]) for k=1:num_conf_i , i=1:num_spins]

        x = learned_gm[current_spin,:]

        if learning_method_name == "RISE"
            iso_obj_deriv(x,k) = (1/num_samples_i)*( sum(samples_i[:,1].*(exp.(-1*(nodal_stat*x))).*(-1*nodal_stat[:,k])) )
        elseif learning_method_name == "RPLE"
            iso_obj_deriv(x,k) = (1/num_samples_i)*( sum(samples_i[:,1].*(1 ./ (1 + exp.(2*(nodal_stat*x)))).*(-2*nodal_stat[:,k])) )
        end

        for ind_spin = 1:num_spins
            grad_value_i_array[ind_spin] = iso_obj_deriv(x,ind_spin)
        end
        grad_value += norm(grad_value_i_array,Inf)
    end
    return grad_value/num_spins
end

# Evaluate the gradient of the objective function afterwards
function get_hessian_value_glauber_dynamics(samples::Array, learned_gm::Array, learning_method::GMLFormulation, learning_method_name::String)
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    grad_value = 0
    hessian_matrix = Array{Float64,1}(num_spins-1,num_spins-1)

    current_spin = rand(1:num_spins)
    samples_i = samples[find(isequal(current_spin),samples[:,2]),:]
    num_conf_i = size(samples_i,1)
    num_samples_i = sum(samples_i[:,1])

    nodal_stat  = [ samples_i[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i[k, 2 + i]) for k=1:num_conf_i , i=1:num_spins]
    cov_stat = [ samples_i[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i[k, 2 + i]) for k=1:num_conf_i , i=1:num_spins]

    x = learned_gm[current_spin,:]

    if learning_method_name == "RISE"
        iso_obj_deriv(x,k,l) = (1/num_samples_i)*( sum(samples_i[:,1].*(exp.(-1*(nodal_stat*x))).*(-1*nodal_stat[:,k])) )
    elseif learning_method_name == "RPLE"
        iso_obj_deriv(x,k,l) = (1/num_samples_i)*( sum(samples_i[:,1].*(1 ./ (1 + exp.(2*(nodal_stat*x)))).*(-2*nodal_stat[:,k])) )
    end

    for ind_spin = 1:num_spins
        grad_value_i_array[ind_spin] = iso_obj_deriv(x,ind_spin)
    end

    return grad_value/num_spins
end
