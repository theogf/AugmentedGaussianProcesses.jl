"""
    train!(model::AbstractGPModel; iterations::Integer=100, callback, convergence)

Function to train the given GP `model`.
## Arguments
- `model` : AbstractGPModel model with either an `Analytic`, `AnalyticVI` or `NumericalVI` type of inference

## Keyword Arguments
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function=nothing` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `convergence::Function=nothing` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(
    model::AbstractGPModel{T},
    X::AbstractArray,
    y,
    iterations::Int=100;
    callback=nothing,
    convergence=nothing,
    state=nothing,
    obsdim=1,
) where {T}
    iterations > 0 || error("Number of iterations should be positive")
    X, Tx = wrap_X(X, obsdim)
    data = wrap_data(X, y, likelihood(model))
    if is_stochastic(model)
        0 < batchsize(inference(model)) <= n_sample(data) || error(
            "The size of mini-batch $(batchsize(inference(model))) is incorrect (negative or bigger than number of samples), please set `batchsize` correctly in the inference object",
        )
        set_ρ!(model, n_sample(data) / batchsize(inference(model)))
    else
        set_batchsize!(inference(model), n_sample(data))
    end

    if verbose(model) > 0
        @info "Starting training $model with $(n_sample(data)) samples, $(n_dim(data)) features and $(n_latent(likelihood(model))) latent GP" *
              (n_latent(model) > 1 ? "s" : "")
    end
    # model.evol_conv = [] # Array to check on the evolution of convergence
    local_iter = 1
    if isnothing(state)
        setHPupdated!(inference(model), true)
    end
    state = isnothing(state) ? init_state(model) : state
    conv = Inf
    p = Progress(iterations; dt=0.2, desc="Training Progress: ")
    prev_elbo = -Inf
    while true # loop until one condition is matched
        try
            if is_stochastic(model)
                minibatch = StatsBase.sample(
                    1:n_sample(data), batchsize(model); replace=false
                )
                x = view_x(data, minibatch)
                y = view_y(likelihood(model), data, minibatch)
            else
                x = input(data)
                y = view_y(likelihood(model), data, 1:n_sample(data))
            end
            state = update_parameters!(model, state, x, y) #Update all the variational parameters
            set_trained!(model, true)
            if !isnothing(callback)
                callback(model, state, n_iter(model)) #Use a callback method if set by user
            end
            if (n_iter(model) % model.atfrequency == 0) &&
               (n_iter(model) >= 3) &&
               (local_iter != iterations)
                state = update_hyperparameters!(model, state, x, y) #Update the hyperparameters
            end
            # Print out informations about the convergence
            if verbose(model) > 2 || (verbose(model) > 1 && local_iter % 10 == 0)
                if inference(model) isa GibbsSampling
                    next!(p; showvalues=[(:samples, local_iter)])
                else
                    if (verbose(model) == 2 && local_iter % 10 == 0)
                        elbo = objective(model, state, y)
                        prev_elbo = elbo
                        ProgressMeter.update!(p, local_iter - 1)
                        ProgressMeter.next!(
                            p; showvalues=[(:iter, local_iter), (:ELBO, elbo)]
                        )
                    elseif verbose(model) > 2
                        elbo = objective(model, state, y)
                        prev_elbo = elbo
                        ProgressMeter.next!(
                            p; showvalues=[(:iter, local_iter), (:ELBO, prev_elbo)]
                        )
                    end
                end
            end
            local_iter += 1
            model.inference.n_iter += 1
            (local_iter <= iterations) || break #Verify if the number of maximum iterations has been reached
        # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
        catch e
            if isa(e, InterruptException)
                @warn "Training interrupted by user at iteration $local_iter"
                break
            else
                rethrow(e)
            end
        end
    end
    if verbose(model) > 0
        @info "Training ended after $(local_iter - 1) iterations. Total number of iterations $(n_iter(model))"
    end
    state = merge(state, compute_Ks(model)) # Compute final version of the matrices for predictions
    post_step!(model, state)
    set_trained!(model, true)
    return model, state
end

@traitfn function train!(
    model::TGP, iterations; callback=nothing, convergence=nothing, state=nothing
) where {TGP <: AbstractGPModel; IsFull{TGP}}
    return train!(
        model,
        input(model.data),
        output(model.data),
        iterations;
        state,
        callback,
        convergence,
    )
end

function update_parameters!(model::GP, state, x, y)
    state = compute_kernel_matrices(model, state, x) # Recompute the matrices if necessary (when hyperparameters have been updated)
    state = analytic_updates(model, state, y)
    return state
end

function update_parameters!(model::VGP, state, x, y)
    state = compute_kernel_matrices(model, state, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = variational_updates(model, state, y)
    return state
end

## Update all variational parameters of the sparse variational GP Model ##
function update_parameters!(model::SVGP, state, x, y)
    state = compute_kernel_matrices(model, state, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = variational_updates(model, state, y)
    return state
end

function update_parameters!(model::MOVGP, state, x, y)
    state = compute_kernel_matrices(model, state, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = update_A!(model, state, y)
    state = variational_updates(model, state, y)
    return state
end

function update_parameters!(m::MOSVGP, state, x, y)
    state = compute_kernel_matrices(m, state, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    update_A!(m, state, y)
    state = variational_updates(m, state, y)
    return state
end

function update_parameters!(m::VStP, state, x, y)
    state = compute_kernel_matrices(m, state, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = local_prior_updates!(m, state, x)
    state = variational_updates(m, state, y)
    return state
end

function compute_kernel_matrices(m::GP{T}, state, x, ::Bool) where {T}
    kernel_matrices = (; K=compute_K(m.f, x, T(jitt)))
    setHPupdated!(inference(m), false)
    return merge(state, (; kernel_matrices))
end

@traitfn function compute_kernel_matrices(
    m::TGP, state, x, update::Bool=false
) where {T,TGP<:AbstractGPModel{T};IsFull{TGP}}
    if isHPupdated(inference(m)) || update
        kernel_matrices = map(m.f) do gp
            (; K=compute_K(gp, x, T(jitt)))
        end
        setHPupdated!(inference(m), false)
        return merge(state, (; kernel_matrices))
    else
        return state
    end
end

@traitfn function compute_kernel_matrices(
    m::TGP, state, x, update::Bool=false
) where {T,TGP<:AbstractGPModel{T};!IsFull{TGP}}
    kernel_matrices = if isHPupdated(inference(m)) || update
        map(m.f) do gp
            (; K=compute_K(gp, T(jitt)))
        end
    else
        state.kernel_matrices
    end
    # If change of hyperparameters or if stochatic
    kernel_matrices = if isHPupdated(inference(m)) || is_stochastic(inference(m)) || update
        κs = map(m.f, kernel_matrices) do gp, kernel_matrix
            compute_κ(gp, x, kernel_matrix.K, T(jitt))
        end
        merge.(kernel_matrices, κs)
    else
        kernel_matrices
    end
    setHPupdated!(inference(m), false)
    return merge(state, (; kernel_matrices)) # update the kernel_matrices state
end

function compute_Ks(m::AbstractGPModel{T}) where {T}
    return (; kernel_matrices=broadcast(m.f, Zviews(m)) do gp, x
        K = compute_K(gp, x, T(jitt))
        return (; K)
    end)
end

function compute_Ks(m::GP{T}) where {T}
    return (; kernel_matrices=(; K=compute_K(m.f, input(m.data), T(jitt))))
end
