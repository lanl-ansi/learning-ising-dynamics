## Defines functions to be used for active learning

# Fetch upper triangle of adj matrix and create parameter vector
# Ref: https://gist.github.com/tpapp/87e5675b87dbdd7a9a840e546eb20fae
function vec_triu_loop(adj_matrix::Array)
    m, n = size(adj_matrix)
    m == n || throw(error("not square"))
    len_theta = n*(n+1) ÷ 2

    # Array to hold values of adjacency matrix
    theta = ones(len_theta)

    # Create array of tuples to hold edges of corresponding values in theta
    edges_theta = Array{Tuple{Int, Int}}(undef, len_theta)

    k = 0
    @inbounds for i in 1:n
        for j in 1:i
            theta[k + j] = adj_matrix[j, i]
            edges_theta[k+j] = (j,i)
        end
        k += i
    end
    return len_theta, theta, edges_theta
end


function remove_node_from_edge(edge::Tuple, node::Integer)
    #=
    Functionality (1,2)\{2} = 1. Note that (1,1)\{1} = 1 as desired here.
    =#
    if edge[1] == node
        return edge[2]
    elseif edge[2] == node
        return edge[1]
    else
        return nothing
    end
end


# Function to calculate the fisher information matrix for a given query
function fisher_information_query(adj_matrix::Array,theta::Array,edges_theta::Array,sigma::Array,len_theta::Integer,num_spins::Integer)
    #=
    Inputs:
    adjacency matrix of the graphical model (estimated) -> corresponds to your theta
    sigma which is the particular query
    =#

    # Define Fisher Information matrix
    I_x = zeros((len_theta, len_theta))

    # Start filling up the upper triangle of I_x
    for a in 1:len_theta
        for b in a:len_theta
            # Get common node if exists
            set_common_nodes_temp = intersect(edges_theta[a],edges_theta[b])

            # if s is empty do nothing (i.e., I_x[a,b] = 0) else
            if !(isempty(set_common_nodes_temp))
                for s in set_common_nodes_temp
                    ind1 = remove_node_from_edge(edges_theta[a],s)
                    ind2 = remove_node_from_edge(edges_theta[b],s)

                    # set shared spin s to 1 as A_s = \sum_{u in ∂s} J_{su} σ_u + J_{ss}
                    sigma_temp = copy(sigma)
                    sigma_temp[s] = 1
                    A_s = sum(sigma_temp[i]*adj_matrix[s,i] for i=1:num_spins)

                    # We pick from sigma_temp instead of sigma for scenarios such as (1,1)∩(1,2)
                    I_x[a,b] += (1.0/num_spins)*sigma_temp[ind1]*sigma_temp[ind2]*(1.0 - tanh(A_s)^2)
                end
            end
        end
    end

    # Now copy over the upper triangle to lower triangle
    D = Diagonal(I_x)
    I_x_sut = I_x - D
    I_x = I_x_sut + transpose(I_x_sut) + D
    return I_x
end


# Function to calculate the fisher information matrix for a given query distribution
function fisher_information_query_distrn(adj_matrix::Array,theta::Array,edges_theta::Array,X_σ::Array,q_σ::Array,len_theta::Integer,num_spins::Integer)
    #=
    Inputs:
    X_σ is the set of queries we are considering
    q_σ is the query distribution over these queries
    =#

    # Make sure the given query distribution is a valid pdf
    #if sum(q_σ)!=1
    #    error("Given query distribution is not a valid pdf")
    #end

    I_q = zeros((len_theta, len_theta))
    for ind_σ=1:length(X_σ)
        sigma0 = 2*digits(X_σ[ind_σ], base=2, pad=num_spins).-1
        I_q += q_σ[ind_σ]*fisher_information_query(adj_matrix,theta,edges_theta,sigma0,len_theta,num_spins)
    end

    return I_q
end


# Function to calculate the fisher information matrix for a given query distribution where q is a Variable (Convex.jl)
function fisher_information_query_distrn_obj(adj_matrix::Array,theta::Array,edges_theta::Array,X_σ::Array,q_σ::Variable,len_theta::Integer,num_spins::Integer)
    #=
    Inputs:
    X_σ is the set of queries we are considering
    q_σ is the query distribution over these queries
    =#

    I_q = zeros((len_theta, len_theta))
    for ind_σ=1:length(X_σ)
        sigma0 = 2*digits(X_σ[ind_σ], base=2, pad=num_spins).-1
        I_q += q_σ[ind_σ]*fisher_information_query(adj_matrix,theta,edges_theta,sigma0,len_theta,num_spins)
    end

    return I_q
end


# Function to calculate the fisher information matrix for a given query distribution
function active_learning_M(adj_matrix::Array,X_σ::Array,num_spins::Integer)
    #=
    Inputs:
    adj_matrix is the current estimate of the parameters
    X_σ is the set of queries we are considering
    q_σ is the query distribution over these queries

    Ref: https://www.juliaopt.org/Convex.jl/dev/types/
    =#

    # Create the parameter vector: θ = [J_11, J_12, J_13, ..., J1n, ..., J_{i,i+1}, ..., J_nn]
    len_theta, theta, edges_theta = vec_triu_loop(adj_matrix)

    # Create the variables required for the SDP
    len_X = length(X_σ)
    q = Variable(len_X)  # query distribution
    t = Variable(len_theta)  # auxillary variables

    # Constraints for the query distribution
    constraints = [q>=0, q<=1, sum(q)==1]

    # Constraints involving the matrices
    U = 1.0*Matrix(I,len_theta,len_theta)
    for i=1:len_theta
        constraints += [fisher_information_query_distrn_obj(adj_matrix, theta, edges_theta, X_σ, q, len_theta, num_spins) U[i,:]; transpose(U[i,:]) t[i]] ⪰ 0
        #constraints += [t[i] transpose(U[i,:]); U[i,:] fisher_information_query_distrn_obj(adj_matrix, theta, edges_theta, X_σ, q, len_theta, num_spins)] ⪰ 0
    end

    # Solve the SDP
    problem = minimize(sum(t), constraints)
    #solve!(problem, SCS.Optimizer)
    solve!(problem, Mosek.Optimizer)

    return q.value
end


# Function to calculate the entropy of a given query
function entropy_query(adj_matrix::Array,sigma::Array,num_spins::Integer)
    #=
    Inputs:
    adjacency matrix of the graphical model (estimated) -> corresponds to your theta
    sigma which is the particular query
    =#

    # Define entropy
    entropy = 0.0

    # Start filling up the upper triangle of I_x
    for k in 1:num_spins
        sigma_temp = copy(sigma)
        sigma_temp[k] = 1
        A_k = sum(sigma_temp[i]*adj_matrix[k,i] for i=1:num_spins)

        entropy += log(2*cosh(A_k)) - A_k*tanh(A_k)
    end

    return entropy
end


# Function to calculate the entropy for a given set of queries/configurations
function entropy_configs(adj_matrix::Array,X_σ::Array,num_spins::Integer)
    #=
    Inputs:
    adjacency matrix of the graphical model (estimated) -> corresponds to your theta
    X_⁠σ which is the set of queries
    =#

    # Define entropy
    entropy_configs = zeros(length(X_σ))

    for i in 1:length(X_σ)
        sigma_temp = 2*digits(X_σ[i], base=2, pad=num_spins) .- 1
        entropy_configs[i] = (1/length(X_σ))*entropy_query(adj_matrix, sigma_temp, num_spins)
    end

    return entropy_configs
end


# Unit test (by visualization) to check if sampling of M-regime with query was done correctly
function unit_test_sampling_M_regime_query(samples_U::Array, spin_number::Integer)
    config_number = 2^spin_number
    spin_configs = []
    for i = 0:(config_number - 1)
        spin_tmp = 2*digits(i, base=2, pad=spin_number).-1
        push!(spin_configs, string(spin_tmp))
    end
    dict_spin_configs = Dict(spin_configs[i+1] => i for i = 0:(config_number-1))

    bins_U_sigma0 = Array{Int}(undef, size(samples_U)[1],2)
    for i=1:size(samples_U)[1]
        sigma0_temp = samples_U[i,3:(n+2)]
        config0_temp = dict_spin_configs[string(sigma0_temp)]
        bins_U_sigma0[i,2] = config0_temp
    end

    bins_U_sigma0[:,1] = copy(samples_U[:,1])

    N_sigma0_U = bins_U_sigma0[:,1]
    config_sigma0_U = bins_U_sigma0[:,2]

    q_σ0 = zeros(config_number)
    for i=1:config_number
        idx_temp = findall(isequal(i-1),config_sigma0_U)
        q_σ0[i] = sum(N_sigma0_U[idx_temp])
    end
    q_σ0 = q_σ0/sum(q_σ0)

    return q_σ0
end
