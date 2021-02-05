export sample, gibbs_sampling, gibbs_sampling2, gibbs_sampling_query, get_sho_samples

export GMSampler, Gibbs

export SamplingRegime, S_regime, T_regime, M_regime

using StatsBase
using LinearAlgebra

abstract type GMSampler end

struct Gibbs <: GMSampler end

abstract type SamplingRegime end

# Time-Series and Histogramming
struct T_regime <: SamplingRegime end

# Multiple Restarts and Histogramming
struct M_regime <: SamplingRegime end

# Time-Series without Histogramming
struct S_regime <: SamplingRegime end

function int_to_spin(int_representation::Int, spin_number::Int)
    spin = 2*digits(int_representation, base=2, pad=spin_number) .- 1
    return spin
end


function weigh_proba(int_representation::Int, adj::Array{T,2}, prior::Array{T,1}) where T <: Real
    spin_number = size(adj,1)
    spins = int_to_spin(int_representation, spin_number)
    return exp(((0.5) * spins' * adj * spins + prior' * spins)[1])
end


bool_to_spin(bool::Int) = 2*bool-1

function weigh_proba(int_representation::Int, adj::Array{T,2}, prior::Array{T,1}, spins::Array{Int,1}) where T <: Real
    digits!(spins, int_representation, base=2)
    spins .= bool_to_spin.(spins)
    return exp((0.5 * spins' * adj * spins + prior' * spins)[1])
end


# assumes second order
function sample_generation_ising(gm::FactorGraph{T}, samples_per_bin::Integer, bins::Int) where T <: Real
    @assert bins >= 1

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector = copy(transpose(diag(adjacency_matrix)))[1,:]

    items   = [i for i in 0:(config_number-1)]
    assignment_tmp = [0 for i in 1:spin_number] # pre allocate assignment memory
    weights = [weigh_proba(i, adjacency_matrix, prior_vector, assignment_tmp) for i in (0:config_number-1)]

    raw_sample = StatsBase.sample(items, StatsBase.Weights(weights), samples_per_bin*bins, ordered=false)
    raw_sample_bins = reshape(raw_sample, bins, samples_per_bin)

    spin_samples = []
    for b in 1:bins
        # Find number of samples of one configuration
        raw_binning = countmap(raw_sample_bins[b,:])

        spin_sample = [ vcat(raw_binning[i], int_to_spin(i, spin_number)) for i in keys(raw_binning)]
        push!(spin_samples, hcat(spin_sample...)')
    end
    return spin_samples
end


function weigh_proba(int_representation::Int, gm::FactorGraph{T}, spins::Array{Int,1}) where T <: Real
    digits!(spins, int_representation, base=2)
    spins .= bool_to_spin.(spins)
    evaluation = sum( weight*prod(spins[i] for i in term) for (term, weight) in gm)
    return exp(evaluation)
end

function sample_generation(gm::FactorGraph{T}, samples_per_bin::Integer, bins::Int) where T <: Real
    @assert bins >= 1
    #info("use general sample model")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    items   = [i for i in 0:(config_number-1)]
    assignment_tmp = [0 for i in 1:spin_number] # pre allocate assignment memory
    weights = [weigh_proba(i, gm, assignment_tmp) for i in (0:config_number-1)]

    raw_sample = StatsBase.sample(items, StatsBase.Weights(weights), samples_per_bin*bins, ordered=false)
    raw_sample_bins = reshape(raw_sample, bins, samples_per_bin)

    spin_samples = []
    for b in 1:bins
        raw_binning = countmap(raw_sample_bins[b,:])
        spin_sample = [ vcat(raw_binning[i], int_to_spin(i, spin_number)) for i in keys(raw_binning)]
        push!(spin_samples, hcat(spin_sample...)')
    end
    return spin_samples
end

## Different methods to create histograms given samples from Glauber dynamics
function hist_countmap(samples_pairs_T::Array, spin_number::Integer, config_number::Integer)
    # Works well for smaller number of spins
    spin_pairs_configs = vec(collect(Iterators.product(1:spin_number,0:(config_number-1),0:(config_number-1))))
    dict_spin_pairs_configs = Dict(spin_pairs_configs[i+1] => i for i = 0:(length(spin_pairs_configs)-1))
    dict_spin_pairs_strings_configs = Dict(dict_spin_pairs_configs[spin_pairs_configs[i+1]] => vcat(collect(spin_pairs_configs[i+1])[1],2*digits(collect(spin_pairs_configs[i+1])[2],2,spin_number) - 1, 2*digits(collect(spin_pairs_configs[i+1])[3],2,spin_number) - 1) for i = 0:(length(spin_pairs_configs)-1))

    samples_pairs_configs_T = Array{Int64}(size(samples_pairs_T,1))
    for i = 1:size(samples_pairs_T,1)
        samples_pairs_configs_T[i] = dict_spin_pairs_configs[tuple(samples_pairs_T[i,1],samples_pairs_T[i,2],samples_pairs_T[i,3])]
    end

    raw_binning_T = countmap(samples_pairs_configs_T)
    samples_T = [ vcat(raw_binning_T[i], dict_spin_pairs_strings_configs[i]) for i in keys(raw_binning_T)]
    samples_T = hcat(samples_T...)'

    return samples_T
end

function countfreq(A)
    #=
    Ref: https://stackoverflow.com/questions/39101839/the-number-of-occurences-of-elements-in-a-vector-julia
    =#
    d = Dict{Array{eltype(A),1}, Int}()
    for i in 1:size(A,1)
        a = A[i,:]
        if a in keys(d)
            d[a] += 1
        else
            d[a] = 1
        end
    end
    return d
end

function hist_countfreq(samples_pairs_T::Array, spin_number::Integer, config_number::Integer)
    # Dictionary over observed configurations with corr frequency
    dict_obs_configs = countfreq(samples_pairs_T)

    samples_T = Array{Int64,2}(undef,length(dict_obs_configs),2+2*spin_number)
    keys_dict_obs_configs = [key for key in keys(dict_obs_configs)]
    for i = 1:length(keys_dict_obs_configs)
        obs_config = keys_dict_obs_configs[i]
        samples_T[i,:] = vcat(dict_obs_configs[obs_config],obs_config[1],2*digits(obs_config[2],base=2,pad=spin_number) .- 1,2*digits(obs_config[3],base=2,pad=spin_number) .- 1)'
    end

    return samples_T
end

function hist_countfreq2(samples_pairs_T::Array, spin_number::Integer, config_number::Integer)
    # Dictionary over observed configurations with corr frequency
    dict_obs_configs = countfreq(samples_pairs_T)

    samples_T = Array{Int64,2}(undef,length(dict_obs_configs),1+spin_number)
    keys_dict_obs_configs = [key for key in keys(dict_obs_configs)]
    for i = 1:length(keys_dict_obs_configs)
        obs_config = keys_dict_obs_configs[i]
        samples_T[i,:] = vcat(dict_obs_configs[obs_config],obs_config)'
    end

    return samples_T
end

function hist_countfreq_glauber_dynamics(samples_pairs_T::Array, spin_number::Integer, config_number::Integer)
    # Dictionary over observed configurations with corr frequency
    dict_obs_configs = countfreq(samples_pairs_T)

    samples_T = Array{Int64,2}(undef,length(dict_obs_configs),2+2*spin_number)
    keys_dict_obs_configs = [key for key in keys(dict_obs_configs)]
    for i = 1:length(keys_dict_obs_configs)
        obs_config = keys_dict_obs_configs[i]
        samples_T[i,:] = vcat(dict_obs_configs[obs_config],obs_config)'
    end

    return samples_T
end

# Create a dictionary out of a histogram
function dictionary_samples_hist(samples_T::Array)
    d = Dict{Array{eltype(samples_T),1}, Int}()
    for i in 1:size(samples_T,1)
        a = samples_T[i,2:end]
        if a in keys(d)
            d[a] += samples_T[i,1]
        else
            d[a] = samples_T[i,1]
        end
    end
    return d
end

# Add two different histograms (note that there may be new rows)
function add_histograms(samples_T1::Array, samples_T2::Array)
    d1 = dictionary_samples_hist(samples_T1)
    d2 = dictionary_samples_hist(samples_T2)

    for a in keys(d2)
        if a in keys(d1)
            d1[a] += d2[a]
        else
            d1[a] = d2[a]
        end
    end

    samples_T = Array{Int64,2}(undef, length(d1), size(samples_T1,2))
    keys_dict_obs_configs = [key for key in keys(d1)]
    for i = 1:length(keys_dict_obs_configs)
        obs_config = keys_dict_obs_configs[i]
        samples_T[i,:] = vcat(d1[obs_config],obs_config)'
    end

    return samples_T
end

# Convert samples to dictionary over nodes
function get_sho_samples(samples_T::Array)
    num_spins = (size(samples_T,2) - 2)
    num_spins = Int(num_spins/2)

    # Create a dictionary over samples_T
    dict_samples_sho = Dict{Int, Array{Int64, 2}}()

    # Node number is the key of the dictionary
    # Should include those which are flips and those which are not
    # For the cases where there are flips, we know node i is being updated because this is single-site dynamics and we see a change
    # Where there are no flips, we don't know which node is being updated so include all possibilities

    # For each dict[key], array should be of form (n_samples, FLAG_flip, \sigma^{(t)}, \sigma^{(t+1)})

    # First get all those samples where there is no flip
    ind_samples_T_no_flip = []
    ind_samples_T_flip = []

    for ind_config = 1:size(samples_T,1)
        if dot(samples_T[ind_config,3:(2+num_spins)],samples_T[ind_config,(3+num_spins):end]) == num_spins
            push!(ind_samples_T_no_flip,ind_config)
        else
            push!(ind_samples_T_flip,ind_config)
        end
    end

    samples_T_flip = samples_T[ind_samples_T_flip,:]
    samples_T_no_flip = samples_T[ind_samples_T_no_flip,:]

    samples_T_no_flip[:,2] .= 0
    # Now create the dictionary
    for current_spin = 1:num_spins
        samples_current_spin = samples_T_flip[findall(isequal(current_spin),samples_T_flip[:,2]),:]
        samples_current_spin[:,2] .= 1
        dict_samples_sho[current_spin] = vcat(samples_current_spin,samples_T_no_flip)
    end

    return dict_samples_sho
end

function gibbs_sampling_ising(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::S_regime) where T <: Real
    @info("using Glauber dynamics v1 to generate samples")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    # Allocate memory for matrix of samples (time-series)
    #spin_samples = Array{Int8, 2}(num_samples, spin_number)
    spin_samples = Array{Int64, 1}(undef, num_samples)

    # Allocate memory for node chosen for updating
    node_selected_samples = Array{Int8,1}(undef, num_samples-1)

    spin_configs = []
    for i = 0:(config_number - 1)
        spin_tmp = 2*digits(i, base = 2, pad = spin_number).-1
        push!(spin_configs, string(spin_tmp))
    end
    dict_spin_configs = Dict(spin_configs[i+1] => i for i = 0:(config_number-1))

    # Start Gibbs sampling

    # generate a random state initially
    sigma = [-1 for i in 1:spin_number]

    sigma_new = deepcopy(sigma)
    #spin_samples[1,:] = sigma
    spin_samples[1] = dict_spin_configs[string(sigma)]

    for ind_step = 1:(num_samples-1)
        i = rand(1:spin_number)
        #i = ((ind_step - 1) % spin_number) + 1

        # Proposed state
        sigma_new = deepcopy(sigma)
        sigma_new[i] = 1

        # Calculate the probability of accepting new state
        num_prob = exp( 2.0*(dot(adj_matrix[i,:],sigma) + prior_vector[i])[1] )
        denom_prob = 1.0 + num_prob
        acceptance_prob = num_prob / denom_prob

        # Accept new state or not
        r = rand()
        if r < acceptance_prob
            sigma = deepcopy(sigma_new)
        else
            sigma_new[i] = -1
            sigma = deepcopy(sigma_new)
        end

        # Update samples
        #spin_samples[ind_step+1,:] = sigma
        spin_samples[ind_step+1] = dict_spin_configs[string(sigma)]

        # Update node selected
        node_selected_samples[ind_step] = copy(i)
    end

    return spin_samples, node_selected_samples
end

function gibbs_sampling_ising(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::T_regime) where T <: Real
    @info("using Glauber dynamics v1 to generate T-regime samples")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    # Allocate memory for matrix of samples (time-series)
    spin_samples = Array{Int64, 1}(undef, num_samples)

    # Allocate memory for node chosen for updating
    node_selected_samples = Array{Int8,1}(undef, num_samples-1)

    spin_configs = []
    for i = 0:(config_number - 1)
        spin_tmp = 2*digits(i, base = 2, pad = spin_number).-1
        push!(spin_configs, string(spin_tmp))
    end
    dict_spin_configs = Dict(spin_configs[i+1] => i for i = 0:(config_number-1))

    # Start Gibbs sampling

    # generate a random state initially
    sigma_conf = rand(0:(config_number - 1))
    sigma = 2*digits(sigma_conf, base=2, pad=spin_number).-1


    sigma_new = deepcopy(sigma)
    #spin_samples[1,:] = sigma
    spin_samples[1] = dict_spin_configs[string(sigma)]

    for ind_step = 1:(num_samples-1)
        i = rand(1:spin_number)
        #i = ((ind_step - 1) % spin_number) + 1

        # Proposed state
        sigma_new = deepcopy(sigma)
        sigma_new[i] = 1

        # Calculate the probability of accepting new state
        num_prob = exp( 2.0*(dot(adj_matrix[i,:],sigma) + prior_vector[i])[1] )
        denom_prob = 1.0 + num_prob
        acceptance_prob = num_prob / denom_prob

        # Accept new state or not
        r = rand()
        if r < acceptance_prob
            sigma = deepcopy(sigma_new)
        else
            sigma_new[i] = -1
            sigma = deepcopy(sigma_new)
        end

        # Update samples
        spin_samples[ind_step+1] = dict_spin_configs[string(sigma)]

        # Update node selected
        node_selected_samples[ind_step] = copy(i)
    end

    ## T Samples
    samples_pairs_T = hcat(node_selected_samples,spin_samples[1:end-1],spin_samples[2:end])
    samples_T = hist_countfreq(samples_pairs_T, spin_number, config_number)

    ## Mixing the T samples
    raw_binning = countmap(spin_samples)
    samples_mixed = [ vcat(raw_binning[i], 2*digits(i, base=2, pad=spin_number).-1) for i in keys(raw_binning)]
    samples_mixed = hcat(samples_mixed...)'

    return samples_T, samples_mixed
end

function gibbs_sampling_ising(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::M_regime) where T <: Real
    @info("using Glauber dynamics v1 to generate M-regime samples")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    # Allocate memory for matrix of samples
    # Arranged as [node selected, \sigma^{(t)}, \sigma^{(t+1)}]
    samples_pairs_T = Array{Int64, 2}(undef, num_samples, 3)

    spin_configs = []
    for i = 0:(config_number - 1)
        spin_tmp = 2*digits(i, base=2, pad=spin_number).-1
        push!(spin_configs, string(spin_tmp))
    end
    dict_spin_configs = Dict(spin_configs[i+1] => i for i = 0:(config_number-1))

    # Start Gibbs Sampling for Multiple Restarts
    for ind_step = 1:num_samples
        # generate a random state initially
        sigma_conf = rand(0:(config_number - 1))
        sigma = 2*digits(sigma_conf, base=2, pad=spin_number).-1

        # Random generation of site to be changed
        i = rand(1:spin_number)
        #i = ((ind_step - 1) % spin_number) + 1

        # Proposed state
        sigma_new = deepcopy(sigma)
        sigma_new[i] = 1

        # Calculate the probability of accepting new state
        num_prob = exp( 2.0*(dot(adj_matrix[i,:],sigma) + prior_vector[i])[1] )
        denom_prob = 1.0 + num_prob
        acceptance_prob = num_prob / denom_prob

        # Accept new state or not
        r = rand()
        if r < acceptance_prob
            sigma = deepcopy(sigma_new)
        else
            sigma_new[i] = -1
            sigma = deepcopy(sigma_new)
        end

        # Update Samples Array
        samples_pairs_T[ind_step,1] = copy(i)
        samples_pairs_T[ind_step,2] = deepcopy(sigma_conf)
        samples_pairs_T[ind_step,3] = dict_spin_configs[string(sigma)]
    end

    ## M Samples
    samples_T = hist_countfreq(samples_pairs_T, spin_number, config_number)

    ## Mixing the M samples
    spin_samples = samples_pairs_T[:,3]
    raw_binning = countmap(spin_samples)
    samples_mixed = [ vcat(raw_binning[i], 2*digits(i, base=2, pad=spin_number).-1) for i in keys(raw_binning)]
    samples_mixed = hcat(samples_mixed...)'

    return samples_T, samples_mixed
end

function gibbs_sampling_ising2(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::S_regime) where T <: Real
    @info("using Glauber dynamics v2 to generate samples")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    assignment_tmp = [0 for i in 1:spin_number] # pre allocate assignment memory

    # Allocate memory for matrix of samples (time-series)
    spin_samples = Array{Int8, 2}(undef, num_samples, spin_number)

    # Allocate memory for node chosen for updating
    node_selected_samples = Array{Int64,1}(undef, num_samples-1)

    # Do not save dictionary over spin_configs as of length config_number

    # Start Gibbs sampling

    # generate a random state initially
    sigma = [-1 for i in 1:spin_number]

    sigma_new = copy(sigma)
    spin_samples[1,:] = sigma

    for ind_step = 1:(num_samples-1)
        i = ((ind_step - 1) % spin_number) + 1

        # Proposed state - flip ith state
        sigma_new = copy(sigma)
        sigma_new[i] = -1*sigma_new[i]

        # prob_sigma = prob_cond_ising(sigma, i, adjacency_matrix, prior_vector)
        # prob_sigma_new = prob_cond_ising(sigma_new, i, adjacency_matrix, prior_vector)

        # Calculate the probability of accepting new state which is also prob_sigma_new/prob_sigma
        acceptance_prob = exp( -2*sigma[i]*(adj_matrix[i,:]'*sigma + prior_vector[i])[1] )

        # Accept new state or not
        transitionProbability = min(1, acceptance_prob)
        if rand() < transitionProbability
            sigma = copy(sigma_new)
        end

        # Update samples
        spin_samples[ind_step+1,:] = sigma

        # Update node selected
        node_selected_samples[ind_step] = copy(i)
    end

    samples = hist_countfreq2(spin_samples, spin_number, config_number)

    return samples, node_selected_samples
end

function gibbs_sampling_ising2(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::T_regime) where T <: Real
    @info("using Glauber dynamics v2 to generate T-regime samples")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    assignment_tmp = [0 for i in 1:spin_number] # pre allocate assignment memory

    # Allocate memory for matrix of samples (time-series)
    spin_samples = Array{Int8, 2}(undef, num_samples, spin_number)

    # Allocate memory for node chosen for updating
    node_selected_samples = Array{Int8,1}(undef, num_samples-1)

    # Not creating dictionary over spin configs as of length config_number

    # Start Gibbs sampling

    # generate a random state initially
    sigma_conf = rand(0:(config_number - 1))
    sigma = 2*digits(sigma_conf, base=2, pad=spin_number).-1
    
    sigma_new = copy(sigma)
    spin_samples[1,:] = copy(sigma)

    for ind_step = 1:(num_samples-1)
        i = ((ind_step - 1) % spin_number) + 1

        # Proposed state - flip ith state
        sigma_new = copy(sigma)
        sigma_new[i] = -1*sigma_new[i]

        # prob_sigma = prob_cond_ising(sigma, i, adjacency_matrix, prior_vector)
        # prob_sigma_new = prob_cond_ising(sigma_new, i, adjacency_matrix, prior_vector)

        # Calculate the probability of accepting new state which is also prob_sigma_new/prob_sigma
        acceptance_prob = exp( -2*sigma[i]*(adj_matrix[i,:]'*sigma + prior_vector[i])[1] )

        # Accept new state or not
        transitionProbability = min(1, acceptance_prob)
        if rand() < transitionProbability
            sigma = copy(sigma_new)
        end

        # Update samples
        spin_samples[ind_step+1,:] = copy(sigma)

        # Update node selected
        node_selected_samples[ind_step] = copy(i)
    end

    # Histogramming for the T-regime
    samples_pairs_T = hcat(node_selected_samples,spin_samples[1:end-1,:],spin_samples[2:end,:])
    samples_T = hist_countfreq_glauber_dynamics(samples_pairs_T, spin_number, config_number)

    # Mixing the T samples
    samples_mixed = hist_countfreq2(spin_samples, spin_number, config_number)

    return samples_T, samples_mixed
end

function gibbs_sampling_ising2(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::M_regime) where T <: Real
    @info("using Glauber dynamics v2 to generate M-regime samples")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    # Allocate memory for matrix of samples
    # Arranged as [node selected, \sigma^{(t)}, \sigma^{(t+1)}]
    samples_pairs_T = Array{Int64, 2}(undef, num_samples, 1 + 2*spin_number)

    # Not creating dictionary over possible configurations as of large size

    # Start Gibbs Sampling for Multiple Restarts
    for ind_step = 1:num_samples
        # generate a random state initially
        sigma_conf = rand(0:(config_number - 1))
        sigma = 2*digits(sigma_conf, base=2, pad=spin_number).-1

        # Random generation of site to be changed
        i = rand(1:spin_number)
        #i = ((ind_step - 1) % spin_number) + 1

        # Proposed state - flip ith state
        sigma_old = deepcopy(sigma)
        sigma_new = deepcopy(sigma)
        sigma_new[i] = -1*sigma_new[i]

        # prob_sigma = prob_cond_ising(sigma, i, adjacency_matrix, prior_vector)
        # prob_sigma_new = prob_cond_ising(sigma_new, i, adjacency_matrix, prior_vector)

        # Calculate the probability of accepting new state which is also prob_sigma_new/prob_sigma
        acceptance_prob = exp( -2*sigma[i]*(adj_matrix[i,:]'*sigma + prior_vector[i])[1] )

        # Accept new state or not
        transitionProbability = min(1, acceptance_prob)
        if rand() < transitionProbability
            sigma = copy(sigma_new)
        end

        # Update Samples Array
        samples_pairs_T[ind_step,1] = copy(i)
        samples_pairs_T[ind_step,2:(spin_number+1)] = deepcopy(sigma_old)
        samples_pairs_T[ind_step,(spin_number+2):end] = deepcopy(sigma)
    end

    # Histogramming for the M-regime
    samples_T = hist_countfreq_glauber_dynamics(samples_pairs_T, spin_number, config_number)

    # Mixing the M samples
    spin_samples = samples_pairs_T[:,(spin_number+2):end]
    samples_mixed = hist_countfreq2(spin_samples, spin_number, config_number)

    return samples_T, samples_mixed
end

function gibbs_sampling_ising2_binning(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::T_regime) where T <: Real
    @info("using Glauber dynamics v2 to generate T-regime samples with binning")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    # Not creating dictionary over spin configs as of length config_number

    # Start Gibbs sampling

    # create bins of samples to look at as the overall number of samples is too large
    max_samples_per_bin = 10000000
    min_bins = div(num_samples,max_samples_per_bin)
    n_bins = copy(min_bins)
    rem_samples = rem(num_samples,max_samples_per_bin)
    if rem_samples > 0
        n_bins += 1
    end
    samples_bin_array = [i <= min_bins ? max_samples_per_bin : rem_samples for i=1:n_bins]

    # generate a random state initially
    sigma_conf = rand(0:(config_number - 1))
    sigma = 2*digits(sigma_conf, base=2, pad=spin_number).-1
    sigma_new = copy(sigma)

    # Some temporary structures
    samples_T = []
    samples_T_temp = []
    samples_mixed = []
    samples_mixed_temp = []

    # One bin at a time -- make sure to reuse the last computed sigma from prev bin to initialize new bin
    for ind_bin = 1:n_bins
        num_samples_bin = samples_bin_array[ind_bin]
        # Allocate memory for matrix of samples (time-series)
        spin_samples = Array{Int8, 2}(undef,num_samples_bin, spin_number)
        spin_samples[1,:] = copy(sigma)

        # Allocate memory for node chosen for updating
        node_selected_samples = Array{Int8,1}(undef,num_samples_bin-1)

        for ind_step = 1:(num_samples_bin-1)
            i = ((ind_step - 1) % spin_number) + 1

            # Proposed state - flip ith state
            sigma_new = copy(sigma)
            sigma_new[i] = -1*sigma_new[i]

            # prob_sigma = prob_cond_ising(sigma, i, adjacency_matrix, prior_vector)
            # prob_sigma_new = prob_cond_ising(sigma_new, i, adjacency_matrix, prior_vector)

            # Calculate the probability of accepting new state which is also prob_sigma_new/prob_sigma
            acceptance_prob = exp( -2*sigma[i]*(adj_matrix[i,:]'*sigma + prior_vector[i])[1] )

            # Accept new state or not
            transitionProbability = min(1, acceptance_prob)
            if rand() < transitionProbability
                sigma = copy(sigma_new)
            end

            # Update samples
            spin_samples[ind_step+1,:] = copy(sigma)

            # Update node selected
            node_selected_samples[ind_step] = copy(i)
        end

        # Histogramming for the T-regime
        samples_pairs_T = hcat(node_selected_samples,spin_samples[1:end-1,:],spin_samples[2:end,:])
        if ind_bin == 1
            samples_T = hist_countfreq_glauber_dynamics(samples_pairs_T, spin_number, config_number)
        else
            samples_T_temp = hist_countfreq_glauber_dynamics(samples_pairs_T, spin_number, config_number)
            samples_T = add_histograms(samples_T_temp, samples_T)
        end

        # Mixing the T samples
        if ind_bin == 1
            samples_mixed = hist_countfreq2(spin_samples, spin_number, config_number)
        else
            samples_mixed_temp = hist_countfreq2(spin_samples, spin_number, config_number)
            samples_mixed = add_histograms(samples_mixed_temp, samples_mixed)
        end
    end

    return samples_T, samples_mixed
end

function gibbs_sampling_ising2_binning(gm::FactorGraph{T}, num_samples::Integer, sampling_regime::M_regime) where T <: Real
    @info("using Glauber dynamics v2 to generate M-regime samples with binning")

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    # Not creating dictionary over spin configs as of length config_number

    # Start Gibbs sampling

    # create bins of samples to look at as the overall number of samples is too large
    max_samples_per_bin = 10000000
    min_bins = div(num_samples,max_samples_per_bin)
    n_bins = copy(min_bins)
    rem_samples = rem(num_samples,max_samples_per_bin)
    if rem_samples > 0
        n_bins += 1
    end
    samples_bin_array = [i <= min_bins ? max_samples_per_bin : rem_samples for i=1:n_bins]

    # generate a random state initially
    sigma_conf = rand(0:(config_number - 1))
    sigma = 2*digits(sigma_conf, base=2, pad=spin_number).-1
    sigma_new = copy(sigma)

    # Some temporary structures
    samples_T = []
    samples_T_temp = []
    samples_mixed = []
    samples_mixed_temp = []

    # One bin at a time -- make sure to reuse the last computed sigma from prev bin to initialize new bin
    for ind_bin = 1:n_bins
        num_samples_bin = samples_bin_array[ind_bin]

        # Allocation memory for samples
        samples_pairs_T = Array{Int64, 2}(undef,num_samples_bin, 1 + 2*spin_number)

        # Allocate memory for node chosen for updating
        node_selected_samples = Array{Int8,1}(undef,num_samples_bin)

        # Start Gibbs Sampling for Multiple Restarts
        for ind_step = 1:num_samples_bin
            # generate a random state initially
            sigma_conf = rand(0:(config_number - 1))
            sigma = 2*digits(sigma_conf, base=2, pad=spin_number)-1

            # Random generation of site to be changed
            i = rand(1:spin_number)
            #i = ((ind_step - 1) % spin_number) + 1

            # Proposed state - flip ith state
            sigma_old = deepcopy(sigma)
            sigma_new = deepcopy(sigma)
            sigma_new[i] = -1*sigma_new[i]

            # prob_sigma = prob_cond_ising(sigma, i, adjacency_matrix, prior_vector)
            # prob_sigma_new = prob_cond_ising(sigma_new, i, adjacency_matrix, prior_vector)

            # Calculate the probability of accepting new state which is also prob_sigma_new/prob_sigma
            acceptance_prob = exp( -2*sigma[i]*(adj_matrix[i,:]'*sigma + prior_vector[i])[1] )

            # Accept new state or not
            transitionProbability = min(1, acceptance_prob)
            if rand() < transitionProbability
                sigma = copy(sigma_new)
            end

            # Update Samples Array
            samples_pairs_T[ind_step,1] = copy(i)
            samples_pairs_T[ind_step,2:(spin_number+1)] = deepcopy(sigma_old)
            samples_pairs_T[ind_step,(spin_number+2):end] = deepcopy(sigma)
        end

        # Histogramming for the M-regime
        if ind_bin == 1
            samples_T = hist_countfreq_glauber_dynamics(samples_pairs_T, spin_number, config_number)
        else
            samples_T_temp = hist_countfreq_glauber_dynamics(samples_pairs_T, spin_number, config_number)
            samples_T = add_histograms(samples_T_temp, samples_T)
        end

        # Mixing the T samples
        spin_samples = samples_pairs_T[:,(spin_number+2):end]
        if ind_bin == 1
            samples_mixed = hist_countfreq2(spin_samples, spin_number, config_number)
        else
            samples_mixed_temp = hist_countfreq2(spin_samples, spin_number, config_number)
            samples_mixed = add_histograms(samples_mixed_temp, samples_mixed)
        end
    end

    return samples_T, samples_mixed
end

sample(gm::FactorGraph{T}, number_sample::Integer) where T <: Real = sample(gm, number_sample, 1, Gibbs())[1]
sample(gm::FactorGraph{T}, number_sample::Integer, replicates::Integer) where T <: Real = sample(gm, number_sample, replicates, Gibbs())


function sample(gm::FactorGraph{T}, number_sample::Integer, replicates::Integer, sampler::Gibbs) where T <: Real
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs, given alphabet $(gm.alphabet)")
    end

    if gm.order <= 2
        samples = sample_generation_ising(gm, number_sample, replicates)
    else
        samples = sample_generation(gm, number_sample, replicates)
    end

    return samples
end

# Function to generate samples according to Glauber dynamics
gibbs_sampling(gm::FactorGraph{T}, number_sample::Integer) where T <: Real = gibbs_sampling(gm, number_sample, S_regime())
gibbs_sampling2(gm::FactorGraph{T}, number_sample::Integer) where T <: Real = gibbs_sampling2(gm, number_sample, S_regime())

function gibbs_sampling(gm::FactorGraph{T}, number_sample::Integer, sampling_regime::SamplingRegime) where T <: Real
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs, given alphabet $(gm.alphabet)")
    end

    if gm.order <= 2
        samples = gibbs_sampling_ising(gm, number_sample, sampling_regime)
    else
        error("not supported yet")
    end

    return samples
end

function gibbs_sampling2(gm::FactorGraph{T}, number_sample::Integer, sampling_regime::SamplingRegime) where T <: Real
    max_num_samples = 10000000

    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs, given alphabet $(gm.alphabet)")
    end

    if gm.order <= 2 && number_sample <= max_num_samples
        samples = gibbs_sampling_ising2(gm, number_sample, sampling_regime)
    elseif gm.order <= 2 && number_sample > max_num_samples
        samples = gibbs_sampling_ising2_binning(gm, number_sample, sampling_regime)
    else
        error("not supported yet")
    end

    return samples
end

# Function to generate samples according to Glauber dynamics and given a query distribution for σ0
function gibbs_sampling_query(gm::FactorGraph{T}, num_samples::Integer, X_σ::Array, q::Array, sampling_regime::SamplingRegime, FLAG_verbose=true) where T <: Real
    #=
    Inputs:
    X_σ is the alphabet of interest (array of integers to be converted to binary strings)
    q is the query distribution
    =#
    if FLAG_verbose
        @info("using Glauber dynamics query sampling v1 to generate M-regime samples")
    end

    # Error messages
    if gm.alphabet != :spin
        error("sampling is only supported for spin FactorGraphs, given alphabet $(gm.alphabet)")
    end

    if gm.order > 2
        error("not supported yet")
    end

    spin_number   = gm.varible_count
    config_number = 2^spin_number

    adjacency_matrix = convert(Array{T,2}, gm)
    prior_vector =  transpose(diag(adjacency_matrix))[1,:]
    adj_matrix = adjacency_matrix - diagm(0 => diag(adjacency_matrix))

    # Allocate memory for matrix of samples
    # Arranged as [node selected, \sigma^{(t)}, \sigma^{(t+1)}]
    samples_pairs_T = Array{Int64, 2}(undef, num_samples, 3)

    spin_configs = []
    for i = 0:(config_number - 1)
        spin_tmp = 2*digits(i, base=2, pad=spin_number).-1
        push!(spin_configs, string(spin_tmp))
    end
    dict_spin_configs = Dict(spin_configs[i+1] => i for i = 0:(config_number-1))

    # Start Gibbs Sampling for Multiple Restarts
    for ind_step = 1:num_samples
        # generate a random state from X_σ according to distrn q
        sigma_conf = StatsBase.sample(X_σ, StatsBase.weights(q))
        sigma = 2*digits(sigma_conf, base=2, pad=spin_number).-1

        # Random generation of site to be changed
        i = rand(1:spin_number)
        #i = ((ind_step - 1) % spin_number) + 1

        # Proposed state
        sigma_new = deepcopy(sigma)
        sigma_new[i] = 1

        # Calculate the probability of accepting new state
        num_prob = exp( 2.0*(dot(adj_matrix[i,:],sigma) + prior_vector[i])[1] )
        denom_prob = 1.0 + num_prob
        acceptance_prob = num_prob / denom_prob

        # Accept new state or not
        r = rand()
        if r < acceptance_prob
            sigma = deepcopy(sigma_new)
        else
            sigma_new[i] = -1
            sigma = deepcopy(sigma_new)
        end

        # Update Samples Array
        samples_pairs_T[ind_step,1] = copy(i)
        samples_pairs_T[ind_step,2] = deepcopy(sigma_conf)
        samples_pairs_T[ind_step,3] = dict_spin_configs[string(sigma)]
    end

    ## M Samples
    samples_T = hist_countfreq(samples_pairs_T, spin_number, config_number)

    ## Mixing the M samples
    spin_samples = samples_pairs_T[:,3]
    raw_binning = countmap(spin_samples)
    samples_mixed = [ vcat(raw_binning[i], 2*digits(i, base=2, pad=spin_number).-1) for i in keys(raw_binning)]
    samples_mixed = hcat(samples_mixed...)'

    return samples_T, samples_mixed
end
