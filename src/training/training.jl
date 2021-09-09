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
) where {T}
    if model.verbose > 0
        @info "Starting training $model with $(nSamples(model)) samples, $(nFeatures(model)) features and $(nLatent(model)) latent GP" *
              (nLatent(model) > 1 ? "s" : "")
    end

    iterations > 0 || error("Number of iterations should be positive")
    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter = 1
    state = isnothing(state) ? init_state(model) : state
    conv = Inf
    p = Progress(iterations; dt=0.2, desc="Training Progress: ")
    prev_elbo = -Inf
    while true # loop until one condition is matched
        try
            state = update_parameters!(model, state) #Update all the variational parameters
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

function update_parameters!(model::GP, state)
    state = compute_kernel_matrices!(model, state) #Recompute the matrices if necessary (when hyperparameters have been updated)
    state = analytic_updates!(model, state)
    return state
end

function update_parameters!(model::VGP, state)
    state = compute_kernel_matrices!(model, state) #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = variational_updates!(model, state)
    return state
end

## Update all variational parameters of the sparse variational GP Model ##
function update_parameters!(m::SVGP, state)
    if isStochastic(inference(m))
        setMBIndices!(
            inference(m),
            StatsBase.sample(1:nSamples(m), nMinibatch(inference(m)); replace=false),
        )
        setxview!(inference(m), view_x(data(m), MBIndices(inference(m))))
        setyview!(inference(m), view_y(likelihood(m), data(m), MBIndices(inference(m))))
    end
    state = compute_kernel_matrices!(m, state) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = variational_updates!(m, state)
    return state
end

function update_parameters!(m::MOVGP, state)
    state = compute_kernel_matrices!(m, state) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = update_A!(m)
    state = variational_updates!(m, state)
    return nothing
end

function update_parameters!(m::MOSVGP)
    if isStochastic(inference(m))
        setMBIndices!(
            inference(m),
            StatsBase.sample(1:nSamples(m), nMinibatch(inference(m)); replace=false),
        )
        setxview!(inference(m), view_x(data(m), MBIndices(inference(m))))
        setyview!(m.inference, view_y(likelihood(m), data(m), MBIndices(m.inference)))
    end
    state = compute_kernel_matrices!(m, state) # Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
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

function compute_kernel_matrices!(m::GP{T}, state, ::Bool) where {T}
    compute_K!(getf(m), input(m), T(jitt))
    setHPupdated!(inference(m), false)
    return state
end

@traitfn function compute_kernel_matrices!(
    m::TGP, state, update::Bool=false
) where {T,TGP<:AbstractGP{T};IsFull{TGP}}
    if isHPupdated(inference(m)) || update
        compute_K!.(getf(m), Ref(input(m)), T(jitt))
    end
    setHPupdated!(inference(m), false)
    return state
end

@traitfn function compute_kernel_matrices!(
    m::TGP, state, update::Bool=false
) where {T,TGP<:AbstractGP{T};!IsFull{TGP}}
    if isHPupdated(inference(m)) || update
        compute_K!.(getf(m), T(jitt))
    end
    #If change of hyperparameters or if stochatic
    if isHPupdated(inference(m)) || isStochastic(inference(m)) || update
        compute_κ!.(getf(m), Ref(xview(m)), T(jitt))
    end
    setHPupdated!(inference(m), false)
    return state
end
