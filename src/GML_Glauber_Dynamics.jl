module GML_Glauber_Dynamics

export learn, inverse_ising
export learn_glauber_dynamics
export learn_glauber_dynamics_sho, learn_glauber_dynamics_sho2

export GMLFormulation, RISE, logRISE, RPLE, RISEA, multiRISE
export GMLMethod, NLP

using JuMP
using Ipopt
using Printf

import LinearAlgebra
import LinearAlgebra: diag
import LinearAlgebra: diagm
import Statistics: mean


include("models.jl")

include("sampling.jl")

abstract type GMLFormulation end

mutable struct multiRISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
    interaction_order::Integer
end
# default values
multiRISE() = multiRISE(0.4, true, 2)

mutable struct RISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RISE() = RISE(0.4, true)

mutable struct RISEA <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RISEA() = RISEA(0.4, true)

mutable struct logRISE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
logRISE() = logRISE(0.8, true)

mutable struct RPLE <: GMLFormulation
    regularizer::Real
    symmetrization::Bool
end
# default values
RPLE() = RPLE(0.2, true)


abstract type GMLMethod end

mutable struct NLP <: GMLMethod
    #solver::JuMP.OptimizerFactory
    # Hard coded the update from the package online
    solver::Any
end
# default values
NLP() = NLP(with_optimizer(Ipopt.Optimizer, print_level=0, tol=1e-12))

# Other Solvers
# Coordinate descent
mutable struct CD <: GMLMethod
    solver_tolerance::Real
end
# default value
CD() = CD(1e-8)

# default settings
learn(samples::Array{T,2}) where T <: Real = learn(samples, RISE(), NLP())
learn(samples::Array{T,2}, formulation::S) where {T <: Real, S <: GMLFormulation} = learn(samples, formulation, NLP())

#TODO add better support for Adjoints
learn(samples::LinearAlgebra.Adjoint, args...) = learn(copy(samples), args...)


function data_info(samples::Array{T,2}) where T <: Real
    (num_conf, num_row) = size(samples)
    num_spins = num_row - 1
    num_samples = sum(samples[1:num_conf,1])
    return num_conf, num_spins, num_samples
end

function data_info_glauber_dynamics(samples::Array{T,2}) where T <: Real
    (num_conf, num_row) = size(samples)
    num_spins = num_row - 2
    num_spins = Int(num_spins/2)
    num_samples = sum(samples[1:num_conf,1])
    return num_conf, num_spins, num_samples
end

function learn(samples::Array{T,2}, formulation::multiRISE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)
    inter_order = formulation.interaction_order

    reconstruction = Dict{Tuple,Real}()

    for current_spin = 1:num_spins
        nodal_stat = Dict{Tuple,Array{Real,1}}()

        for p = 1:inter_order
                nodal_keys = Array{Tuple{},1}()
                neighbours = [i for i=1:num_spins if i!=current_spin]
                if p == 1
                    nodal_keys = [(current_spin,)]
                else
                    perm = permutations(neighbours, p - 1)
                    if length(perm) > 0
                        nodal_keys = [(current_spin, perm[i]...) for i=1:length(perm)]
                    end
                end

                for index = 1:length(nodal_keys)
                    nodal_stat[nodal_keys[index]] =  [ prod(samples[k, 1 + i] for i=nodal_keys[index]) for k=1:num_conf]
                end
        end

        model = Model(method.solver)

        @variable(model, x[keys(nodal_stat)])
        @variable(model, z[keys(nodal_stat)])

        @NLobjective(model, Min,
            sum((samples[k,1]/num_samples)*exp(-sum(x[inter]*stat[k] for (inter,stat) = nodal_stat)) for k=1:num_conf) +
            lambda*sum(z[inter] for inter = keys(nodal_stat) if length(inter)>1)
        )

        for inter in keys(nodal_stat)
            @constraint(model, z[inter] >=  x[inter]) #z_plus
            @constraint(model, z[inter] >= -x[inter]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED

        nodal_reconstruction = JuMP.value.(x)
        for inter = keys(nodal_stat)
            reconstruction[inter] = deepcopy(nodal_reconstruction[inter])
        end
    end

    if formulation.symmetrization
        reconstruction_list = Dict{Tuple,Vector{Real}}()
        for (k,v) in reconstruction
            key = tuple(sort([i for i in k])...)
            if !haskey(reconstruction_list, key)
                reconstruction_list[key] = Vector{Real}()
            end
            push!(reconstruction_list[key], v)
        end

        reconstruction = Dict{Tuple,Real}()
        for (k,v) in reconstruction_list
            reconstruction[k] = mean(v)
        end
    end

    return FactorGraph(inter_order, num_spins, :spin, reconstruction)
end

function learn(samples::Array{T,2}, formulation::RISE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        model = Model(method.solver)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])

        @NLobjective(model, Min,
            sum((samples[k,1]/num_samples)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

function risea_obj(var, stat, weight)
    (num_conf, num_spins) = size(stat)
    #chvar = cosh.(var)
    #shvar = sinh.(var)
    #return sum(weight[k]*prod(chvar[i] - shvar[i]*stat[k,i] for i=1:num_spins) for k=1:num_conf)
    return sum(weight[k]*exp(-sum(var[i]*stat[k,i] for i=1:num_spins)) for k=1:num_conf)
end

function grad_risea_obj(g, var, stat, weight)
    (num_conf, num_spins) = size(stat)
    #chvar = cosh.(var)
    #shvar = sinh.(var)
    #partial_obj = [- weight[k] * prod(chvar[i] - shvar[i]*stat[k,i] for i=1:num_spins) for k=1:num_conf]
    partial_obj = [- weight[k]*exp(-sum(var[i]*stat[k,i] for i=1:num_spins)) for k=1:num_conf]
    for i=1:num_spins
        g[i] = sum(stat[k,i]*partial_obj[k] for k=1:num_conf)
    end
end

function learn(samples::Array{T,2}, formulation::RISEA, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]
        weight = samples[1:num_conf,1] / num_samples

        obj(x...) = risea_obj(x, nodal_stat, weight)
        function grad(g, x...)
            grad_risea_obj(g, x, nodal_stat, weight)
        end

        function l1norm(z...)
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        end

        model = Model(method.solver)

        JuMP.register(model, :obj, num_spins, obj, grad)
        JuMP.register(model, :l1norm, num_spins, l1norm, autodiff=true)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])


        JuMP.setNLobjective(model, :Min, Expr(:call, :+,
                                            Expr(:call, :obj, x...),
                                            Expr(:call, :l1norm, z...)
                                        )
                            )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end


function learn(samples::Array{T,2}, formulation::logRISE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        model = Model(method.solver)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])

        @NLobjective(model, Min,
            log(sum((samples[k,1]/num_samples)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf)) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end


function learn(samples::Array{T,2}, formulation::RPLE, method::NLP) where T <: Real
    num_conf, num_spins, num_samples = data_info(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        nodal_stat  = [ samples[k, 1 + current_spin] * (i == current_spin ? 1 : samples[k, 1 + i]) for k=1:num_conf , i=1:num_spins]

        model = Model(method.solver)

        @variable(model, x[1:num_spins])
        @variable(model, z[1:num_spins])

        @NLobjective(model, Min,
            sum((samples[k,1]/num_samples)*log(1 + exp(-2*sum(x[i]*nodal_stat[k,i] for i=1:num_spins))) for k=1:num_conf) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(model, z[j] >=  x[j]) #z_plus
            @constraint(model, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(model)
        @assert JuMP.termination_status(model) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

## Solvers for learning Glauber Dynamics

# default settings in general
learn_glauber_dynamics(samples::Array{T,2}) where T <: Real = learn_glauber_dynamics(samples, RISE(), NLP())
learn_glauber_dynamics(samples::Array{T,2}, formulation::S) where {T <: Real, S <: GMLFormulation} = learn_glauber_dynamics(samples, formulation, NLP())

# default setting for coordinate descent
learn_glauber_dynamics(samples::Array{T,2}, formulation::S, method::M) where {T <: Real, S <: GMLFormulation, M <: GMLMethod} = learn_glauber_dynamics(samples, RISE(), CD())

## Using the JuMP Solvers
function learn_glauber_dynamics(samples::Array{T,2}, formulation::RISE, method::NLP) where T <: Real
    @info("using JuMP for RISE to learn Glauber dynamics")

    # samples are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        samples_current_spin = samples[findall(isequal(current_spin),samples[:,2]),:]
        num_conf_current_spin = size(samples_current_spin,1)
        num_samples_current_spin = sum(samples_current_spin[:,1])

        #display(samples_current_spin)

        nodal_stat  = [ samples_current_spin[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_current_spin[k, 2 + i]) for k=1:num_conf_current_spin , i=1:num_spins]

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_current_spin[k,1]/num_samples_current_spin)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf_current_spin) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

function learn_glauber_dynamics(samples::Array{T,2}, formulation::logRISE, method::NLP) where T <: Real
    @info("using JuMP for logRISE")

    # samples are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        samples_current_spin = samples[find(isequal(current_spin),samples[:,2]),:]
        num_conf_current_spin = size(samples_current_spin,1)
        num_samples_current_spin = sum(samples_current_spin[:,1])

        #display(samples_current_spin)

        nodal_stat  = [ samples_current_spin[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_current_spin[k, 2 + i]) for k=1:num_conf_current_spin , i=1:num_spins]

        m = Model(solver = method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            log(sum((samples_current_spin[k,1]/num_samples_current_spin)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf_current_spin)) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        status = solve(m)
        @assert status == :Optimal
        reconstruction[current_spin,1:num_spins] = deepcopy(getvalue(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

function learn_glauber_dynamics(samples::Array{T,2}, formulation::RPLE, method::NLP) where T <: Real
    @info("using JuMP for RPLE to learn Glauber dynamics")

    # samples are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        samples_current_spin = samples[findall(isequal(current_spin),samples[:,2]),:]
        num_conf_current_spin = size(samples_current_spin,1)
        num_samples_current_spin = sum(samples_current_spin[:,1])

        #display(samples_current_spin)

        nodal_stat  = [ samples_current_spin[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_current_spin[k, 2 + i]) for k=1:num_conf_current_spin , i=1:num_spins]

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_current_spin[k,1]/num_samples_current_spin)*log(1 + exp(-2*sum(x[i]*nodal_stat[k,i] for i=1:num_spins))) for k=1:num_conf_current_spin) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end


function learn_glauber_dynamics_regularization(samples::Array{T,2}, formulation::RPLE, method::NLP) where T <: Real
    @info("using JuMP for RPLE with right regularization to learn Glauber dynamics")

    # samples are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        samples_current_spin = samples[findall(isequal(current_spin),samples[:,2]),:]
        num_conf_current_spin = size(samples_current_spin,1)
        num_samples_current_spin = sum(samples_current_spin[:,1])

        lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples_current_spin)

        nodal_stat  = [ samples_current_spin[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_current_spin[k, 2 + i]) for k=1:num_conf_current_spin , i=1:num_spins]

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_current_spin[k,1]/num_samples_current_spin)*log(1 + exp(-2*sum(x[i]*nodal_stat[k,i] for i=1:num_spins))) for k=1:num_conf_current_spin) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

function learn_glauber_dynamics_regularization(samples::Array{T,2}, formulation::RISE, method::NLP) where T <: Real
    @info("using JuMP for RISE with right regularization to learn Glauber dynamics")
    @printf("c=%f\n", formulation.regularizer)

    # samples are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        samples_current_spin = samples[findall(isequal(current_spin),samples[:,2]),:]
        num_conf_current_spin = size(samples_current_spin,1)
        num_samples_current_spin = sum(samples_current_spin[:,1])

        lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples_current_spin)

        nodal_stat  = [ samples_current_spin[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_current_spin[k, 2 + i]) for k=1:num_conf_current_spin , i=1:num_spins]

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_current_spin[k,1]/num_samples_current_spin)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf_current_spin) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end


## Coordinate Descent Solvers for learning Glauber Dynamics (using analytical solution)
function learn_glauber_dynamics(samples::Array{T,2}, formulation::RISE, method::CD) where T <: Real
    @info("using coordinate descent solver for RISE to learn Glauber dyamics")

    # samples are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_conf, num_spins, num_samples = data_info_glauber_dynamics(samples)

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(num_spins, num_spins)

    # ISO Objective function
    RISEobjective(x) = exp(-x)

    for current_spin = 1:num_spins
        samples_current_spin = samples[find(isequal(current_spin),samples[:,2]),:]
        num_conf_current_spin = size(samples_current_spin,1)
        num_samples_current_spin = sum(samples_current_spin[:,1])

        conf_weight     = [samples_current_spin[k,1]/num_samples_current_spin for k=1:num_conf_current_spin]
        nodal_stat  = [ samples_current_spin[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_current_spin[k, 2 + i]) for k=1:num_conf_current_spin , i=1:num_spins]

        # Initialize
        x = zeros(num_spins)    # Solution vector
        z = zeros(num_conf)     # Current sum for each sample in histogram

        func_new = sum(conf_weight[k]*(RISEobjective(z[k]))  for k=1:num_conf_current_spin) + lambda*sum(x[j] for j=1:num_spins if current_spin!=j)
        func_old = copy(func_new)

        # Counter for number of iterations
        n_iter = 0

        # Coordinate Descent Method
        diff_func_val = 1
        while diff_func_val > method.solver_tolerance
            # Choosing a cyclic variant
            cycle_updates = randperm(num_spins)

            # Update function value on should
            func_old = copy(func_new)

            for ind_spin = 1:num_spins
                x_old = copy(x)
                func_old = copy(func_new)
                coord = cycle_updates[ind_spin]

                a = sum(conf_weight[k]*(RISEobjective(z[k])/RISEobjective(x[coord]*nodal_stat[k,coord]))  for k=1:num_conf_current_spin)
                b = sum(conf_weight[k]*(RISEobjective(z[k])/RISEobjective(x[coord]*nodal_stat[k,coord]))*nodal_stat[k,coord]  for k=1:num_conf_current_spin)

                epsilon = b/a
                mu = (coord==current_spin ? 0.0 : lambda/a)

                x_update = RISE_soft_thresolding(epsilon, mu)
                z = [ z[k] + (x_update - x[coord])*nodal_stat[k,coord] for k=1:num_conf_current_spin ]
                x[coord] = x_update
            end
            # Update counter and function value
            n_iter = n_iter + 1
            func_new = sum(conf_weight[k]*(RISEobjective(z[k]))  for k=1:num_conf_current_spin) + lambda*sum(x[j] for j=1:num_spins if current_spin!=j)

            diff_func_val = abs(func_new - func_old)
            #diff_func_val = norm(grad_k,1)
        end
        # Asserting solution is optimal?

        reconstruction[current_spin,1:num_spins] = deepcopy(x)
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

## Solvers for learning Glauber Dynamics from Spin-History-Only samples
# default settings
learn_glauber_dynamics_sho(samples::Dict{Int,Array{T,2}}, num_samples::Int) where T <: Real = learn_glauber_dynamics_sho(samples, num_samples, RISE(), NLP())
learn_glauber_dynamics_sho(samples::Dict{Int,Array{T,2}}, num_samples::Int, formulation::S) where {T <: Real, S <: GMLFormulation} = learn_glauber_dynamics_sho(samples, num_samples, formulation, NLP())

function learn_glauber_dynamics_sho(samples::Dict{Int,Array{T,2}}, num_samples::Int, formulation::RISE, method::NLP) where T <: Real
    @info("using JuMP for RISE to learn Glauber dynamics from SHO samples")

    # Dictonary over samples
    # samples[1] are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_spins = length(keys(samples))

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        samples_current_spin = samples[current_spin]
        samples_current_spin_flip = samples_current_spin[findall(isequal(1),samples_current_spin[:,2]),:]

        num_conf_current_spin_flip = size(samples_current_spin_flip,1)
        num_samples_current_spin_flip = sum(samples_current_spin_flip[:,1])

        #display(samples_current_spin)

        nodal_stat  = [ samples_current_spin_flip[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_current_spin_flip[k, 2 + i]) for k=1:num_conf_current_spin_flip , i=1:num_spins]

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_current_spin_flip[k,1]/num_samples_current_spin_flip)*exp(-sum(x[i]*nodal_stat[k,i] for i=1:num_spins)) for k=1:num_conf_current_spin_flip) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

function learn_glauber_dynamics_sho(samples::Dict{Int,Array{T,2}}, num_samples::Int, formulation::RPLE, method::NLP) where T <: Real
    @info("using JuMP for RPLE to learn Glauber dynamics from SHO samples")

    # Dictonary over samples
    # samples[1] are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_spins = length(keys(samples))

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        # Using i in variable names to denote current spin
        samples_i = samples[current_spin]
        samples_i_flip = samples_i[findall(isequal(1),samples_i[:,2]),:]
        samples_i_no_flip = samples_i[findall(isequal(0),samples_i[:,2]),:]

        num_conf_i_flip = size(samples_i_flip,1)
        num_samples_i_flip = sum(samples_i_flip[:,1])

        num_conf_i_no_flip = size(samples_i_no_flip,1)
        num_samples_i_no_flip = sum(samples_i_no_flip[:,1])

        nodal_stat_flip  = [ samples_i_flip[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i_flip[k, 2 + i]) for k=1:num_conf_i_flip , i=1:num_spins]
        nodal_stat_no_flip  = [ samples_i_no_flip[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i_no_flip[k, 2 + i]) for k=1:num_conf_i_no_flip , i=1:num_spins]

        # Probability that i was chosen for updating
        gamma_i = 1.0/num_spins

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_i_flip[k,1]/num_samples_i_flip)*log(1 + exp(-2*sum(x[i]*nodal_stat_flip[k,i] for i=1:num_spins))) for k=1:num_conf_i_flip) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

learn_glauber_dynamics_sho2(samples::Dict{Int,Array{T,2}}, num_samples::Int) where T <: Real = learn_glauber_dynamics_sho2(samples, num_samples, RISE(), NLP())
learn_glauber_dynamics_sho2(samples::Dict{Int,Array{T,2}}, num_samples::Int, formulation::S) where {T <: Real, S <: GMLFormulation} = learn_glauber_dynamics_sho2(samples, num_samples, formulation, NLP())

function learn_glauber_dynamics_sho2(samples::Dict{Int,Array{T,2}}, num_samples::Int, formulation::RISE, method::NLP) where T <: Real
    @info("using JuMP for RISE to learn Glauber dynamics from SHO samples v2")

    # Dictonary over samples
    # samples[1] are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_spins = length(keys(samples))

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        # Using i in variable names to denote current spin
        samples_i = samples[current_spin]
        samples_i_flip = samples_i[findall(isequal(1),samples_i[:,2]),:]
        samples_i_no_flip = samples_i[findall(isequal(0),samples_i[:,2]),:]

        num_conf_i_flip = size(samples_i_flip,1)
        num_samples_i_flip = sum(samples_i_flip[:,1])

        num_conf_i_no_flip = size(samples_i_no_flip,1)
        num_samples_i_no_flip = sum(samples_i_no_flip[:,1])

        nodal_stat_flip  = [ samples_i_flip[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i_flip[k, 2 + i]) for k=1:num_conf_i_flip , i=1:num_spins]
        nodal_stat_no_flip  = [ samples_i_no_flip[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i_no_flip[k, 2 + i]) for k=1:num_conf_i_no_flip , i=1:num_spins]

        # Probability that i was chosen for updating
        gamma_i = 1.0/num_spins

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_i_flip[k,1]/num_samples_i_flip)*exp(-sum(x[i]*nodal_stat_flip[k,i] for i=1:num_spins)) for k=1:num_conf_i_flip) +
            sum((samples_i_no_flip[k,1]/num_samples_i_no_flip)*(gamma_i*exp(-sum(x[i]*nodal_stat_no_flip[k,i] for i=1:num_spins)) + (1.0-gamma_i)) for k=1:num_conf_i_no_flip) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

function learn_glauber_dynamics_sho2(samples::Dict{Int,Array{T,2}}, num_samples::Int, formulation::RPLE, method::NLP) where T <: Real
    @info("using JuMP for RPLE to learn Glauber dynamics from SHO samples v2")

    # Dictonary over samples
    # samples[1] are in the form of [num_samples matching config, config = node selected for update at t, \sigma^{t}, \sigma^{t+1}]
    num_spins = length(keys(samples))

    lambda = formulation.regularizer*sqrt(log((num_spins^2)/0.05)/num_samples)

    reconstruction = Array{Float64}(undef, num_spins, num_spins)

    for current_spin = 1:num_spins
        # Using i in variable names to denote current spin
        samples_i = samples[current_spin]
        #samples_i_flip = samples_i[find(isequal(1),samples_i[:,2]),:]
        samples_i_no_flip = samples_i[findall(isequal(0),samples_i[:,2]),:]

        #num_conf_i_flip = size(samples_i_flip,1)
        #num_samples_i_flip = sum(samples_i_flip[:,1])

        num_conf_i_no_flip = size(samples_i_no_flip,1)
        num_samples_i_no_flip = sum(samples_i_no_flip[:,1])

        num_conf_i = size(samples_i,1)
        num_samples_i = sum(samples_i[:,1])

        nodal_stat  = [ samples_i[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i[k, 2 + i]) for k=1:num_conf_i, i=1:num_spins]
        #nodal_stat_flip  = [ samples_i_flip[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i_flip[k, 2 + i]) for k=1:num_conf_i_flip , i=1:num_spins]
        nodal_stat_no_flip  = [ samples_i_no_flip[k, 2 + num_spins + current_spin] * (i == current_spin ? 1 : samples_i_no_flip[k, 2 + i]) for k=1:num_conf_i_no_flip , i=1:num_spins]

        # Probability that i was chosen for updating
        gamma_i = 1.0/num_spins

        m = Model(method.solver)

        @variable(m, x[1:num_spins])
        @variable(m, z[1:num_spins])

        @NLobjective(m, Min,
            sum((samples_i[k,1]/num_samples_i)*log(1 + exp(-2*sum(x[i]*nodal_stat[k,i] for i=1:num_spins))) for k=1:num_conf_i) -
            #sum((samples_i_no_flip[k,1]/num_samples_i_no_flip)*log(1 + (1.0-gamma_i)*exp(-2*sum(x[i]*nodal_stat_no_flip[k,i] for i=1:num_spins))) for k=1:num_conf_i_no_flip) +
            sum((samples_i_no_flip[k,1]/num_samples_i_no_flip)*(1.0-gamma_i)*log(1 + exp(-2*sum(x[i]*nodal_stat_no_flip[k,i] for i=1:num_spins))) for k=1:num_conf_i_no_flip) +
            lambda*sum(z[j] for j=1:num_spins if current_spin!=j)
        )

        for j in 1:num_spins
            @constraint(m, z[j] >=  x[j]) #z_plus
            @constraint(m, z[j] >= -x[j]) #z_minus
        end

        JuMP.optimize!(m)
        @assert JuMP.termination_status(m) == JuMP.MOI.LOCALLY_SOLVED
        reconstruction[current_spin,1:num_spins] = deepcopy(JuMP.value.(x))
    end

    if formulation.symmetrization
        reconstruction = 0.5*(reconstruction + transpose(reconstruction))
    end

    return reconstruction
end

end
