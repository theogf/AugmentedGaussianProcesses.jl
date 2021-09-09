"""
    train!(model::AbstractGP; iterations::Integer=100, callback, convergence)

Function to train the given GP `model`.
## Arguments
- `model` : AbstractGP model with either an `Analytic`, `AnalyticVI` or `NumericalVI` type of inference

## Keyword Arguments
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function=nothing` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `convergence::Function=nothing` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(
    model::AbstractGP{T},
    X,
    y,
    iterations::Int=100;
    callback=nothing,
    convergence=nothing,
    state=nothing,
    obsdim,
) where {T}
    iterations > 0 || error("Number of iterations should be positive")
    X, T = wrap_X(X, obsdim)
    y, n_latent, likelihood = check_data!(y, likelihood)
    data = wrap_data(X, y)
    if verbose(model) > 0
        @info "Starting training $model with $(n_samples(data)) samples, $(n_features(data)) features and $(n_latent) latent GP" *
              (n_latent(model) > 1 ? "s" : "")
    end
    # model.evol_conv = [] # Array to check on the evolution of convergence
    local_iter = 1
    state = isnothing(state) ? init_state(model) : state
    conv = Inf
    p = Progress(iterations; dt=0.2, desc="Training Progress: ")
    prev_elbo = -Inf
    while true # loop until one condition is matched
        try
            if is_stochastic(inference(m))
                minibatch = StatsBase.sample(
                    1:n_sample(data), batchsize(inference(m)); replace=false
                )
                x = view_x(data, minibatch)
                y = view_y(likelihood(m), data, minibatch)
            else
                x = view_x(data, 1:n_sample(data))
                y = view_y(likelihood(m), data, 1:n_sample(data))
            end
            state = update_parameters!(model, x, y, state) #Update all the variational parameters
            set_trained!(model, true)
            if !isnothing(callback)
                callback(model, state, nIter(model)) #Use a callback method if set by user
            end
            if (nIter(model) % model.atfrequency == 0) &&
               (nIter(model) >= 3) &&
               (local_iter != iterations)
                state = update_hyperparameters!(model, state) #Update the hyperparameters
            end
            # Print out informations about the convergence
            if verbose(model) > 2 || (verbose(model) > 1 && local_iter % 10 == 0)
                if inference(model) isa GibbsSampling
                    next!(p; showvalues=[(:samples, local_iter)])
                else
                    if (verbose(model) == 2 && local_iter % 10 == 0)
                        elbo = objective(model, state)
                        prev_elbo = elbo
                        ProgressMeter.update!(p, local_iter - 1)
                        ProgressMeter.next!(
                            p; showvalues=[(:iter, local_iter), (:ELBO, elbo)]
                        )
                    elseif verbose(model) > 2
                        elbo = objective(model, state)
                        prev_elbo = elbo
                        ProgressMeter.next!(
                            p; showvalues=[(:iter, local_iter), (:ELBO, prev_elbo)]
                        )
                    end
                end
            end
            local_iter += 1
            model.inference.nIter += 1
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
        @info "Training ended after $(local_iter - 1) iterations. Total number of iterations $(nIter(model))"
    end
    compute_kernel_matrices!(model, true) # Compute final version of the matrices for predictions
    post_step!(model)
    set_trained!(model, true)
    return nothing
end

function update_parameters!(model::GP, state, x, y)
    state = compute_kernel_matrices(model, state, x) # Recompute the matrices if necessary (when hyperparameters have been updated)
    state = analytic_updates(model, state, y)
    return state
end

function update_parameters!(model::VGP, state, x, y)
    state = compute_kernel_matrices(model, state, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = variational_updates(state, model, y)
    return state
end

## Update all variational parameters of the sparse variational GP Model ##
function update_parameters!(model::SVGP, state, x, y)
    state = compute_kernel_matrices(model, state, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = variational_updates(model, state, y)
    return state
end

function update_parameters!(m::MOVGP, x, y, state)
    state = compute_kernel_matrices(state, m, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = update_A!(m)
    state = variational_updates!(m, state)
    return nothing
end

function update_parameters!(m::MOSVGP, x, y, state)
    state = compute_kernel_matrices(state, m, x) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    update_A!(m)
    state = variational_updates!(m, state)
    return state
end

function update_parameters!(m::VStP, state)
    state = compute_kernel_matrices!(m, state) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = local_prior_updates!(m, input(m), state)
    state = variational_updates!(m, state)
    return state
end

function compute_kernel_matrices(m::GP{T}, state, x, ::Bool) where {T}
    kernel_matrices = map(getf(m), Ref(x)) do gp, x
        compute_K(gp, x)
    end
    setHPupdated!(inference(m), false)
    return merge(state, (; kernel_matrices))
end

@traitfn function compute_kernel_matrices(
    m::TGP, state, x, update::Bool=false
) where {T,TGP<:AbstractGP{T};IsFull{TGP}}
    if isHPupdated(inference(m)) || update
        kernel_matrices = map(getf(m), Ref(x)) do gp, x
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
) where {T,TGP<:AbstractGP{T};!IsFull{TGP}}
    kernel_matrices = state.kernel_matrices
    kernel_matrices = if isHPupdated(inference(m)) || update
        Ks = map(getf(m)) do gp
            (; K=compute_K(gp, T(jitt)))
        end
        merge.(kernel_matrices, Ks)
    else
        kernel_matrices
    end
    #If change of hyperparameters or if stochatic
    kernel_matrices = if isHPupdated(inference(m)) || isStochastic(inference(m)) || update
        κs = map(getf(m), Ref(x), Ks) do gp, x, K
            compute_κ(gp, K, x, jitt)
        end
        merge.(kernel_matrices, κs)
    else
        kernel_matrices
    end
    setHPupdated!(inference(m), false)
    return merge(state, (; kernel_matrices)) # update the kernel_matrices state
end
