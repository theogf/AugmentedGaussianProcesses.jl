""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,convergence=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `convergence::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(model::AbstractGP{T,TLike,TInf},iterations::Int=100;callback::Union{Nothing,Function}=nothing,convergence::Union{Nothing,Function}=nothing) where {T,TLike<:Likelihood,TInf<:Inference}
    if model.verbose > 0
      println("Starting training $model with $(model.nSamples) samples, $(size(model.X,2)) features and $(model.nLatent) latent GP"*(model.nLatent > 1 ? "s" : ""))
    end

    @assert iterations > 0  "Number of iterations should be positive"
    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter = 1; conv = Inf;
    p = Progress(iterations,dt=0.2,desc="Training Progress: ")
    prev_elbo = -Inf
    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            update_parameters!(model) #Update all the variational parameters
            model.Trained = true
            if !isnothing(callback)
                callback(model,model.inference.nIter) #Use a callback method if set by user
            end
            if (model.inference.nIter%model.atfrequency == 0) && model.inference.nIter >= 3
                update_hyperparameters!(model) #Update the hyperparameters
            end
            # Print out informations about the convergence
            if model.verbose > 2 || (model.verbose > 1  && local_iter%10==0)
                if isa(TInf,GibbsSampling)
                    next!(p; showvalues = [(:samples, local_iter)])
                else
                    if (model.verbose > 2  || local_iter%10==0)
                        elbo = ELBO(model)
                        prev_elbo = elbo
                        next!(p; showvalues = [(:iter, local_iter),(:ELBO,elbo)])
                    else
                        next!(p; showvalues = [(:iter, local_iter),(:ELBO,prev_elbo)])
                    end
                end
            end
            local_iter += 1; model.inference.nIter += 1
            (local_iter <= iterations) || break; #Verify if the number of maximum iterations has been reached
            # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
        catch e
            if isa(e,InterruptException)
                println("Training interrupted by user at iteration $local_iter");
                break;
            else
                rethrow(e)
            end
        end
    end
    if model.verbose > 0
      println("Training ended after $(local_iter-1) iterations. Total number of iterations $(model.inference.nIter)")
    end
    computeMatrices!(model) #Compute final version of the matrices for prediction
    post_process!(model)
    return model.Trained = true
end

function sample(model::MCGP{T,TLike,TInf},nSamples::Int=1000;callback::Union{Nothing,Function}=nothing,cat_samples::Bool=false) where {T,TLike<:Likelihood,TInf<:Inference}
    if model.verbose > 0
      println("Starting sampling $model with $(model.nSamples) samples with $(size(model.X,2)) features and $(model.nLatent) latent GP"*(model.nLatent > 1 ? "s" : ""))
    end
    @assert nSamples > 0  "Number of samples should be positive"
    return sample_parameters(model,nSamples,callback,cat_samples)
end

function update_parameters!(model::GP)
    analytic_updates!(model)
    computeMatrices!(model); #Recompute the matrices if necessary (when hyperparameters have been updated)
end

function update_parameters!(model::VGP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end

function update_parameters!(model::SVGP)
    if model.inference.Stochastic
        model.inference.MBIndices .= StatsBase.sample(1:model.inference.nSamples,model.inference.nMinibatch,replace=false)
        model.inference.xview = view(model.X,model.inference.MBIndices,:)
        model.inference.yview = view_y(model.likelihood,model.y,model.inference.MBIndices)
    end
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end

function update_parameters!(model::MOSVGP)
    if model.inference.Stochastic
        model.inference.MBIndices .= StatsBase.sample(1:model.inference.nSamples,model.inference.nMinibatch,replace=false)
        model.inference.xview = view(model.X,model.inference.MBIndices,:)
        model.inference.yview = view_y.(model.likelihood,model.y,[model.inference.MBIndices])
    end
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    update_A!(model)
    variational_updates!(model);
end

function update_parameters!(model::MOARGP)
    if model.inference.Stochastic
        model.inference.MBIndices .= StatsBase.sample(1:model.inference.nSamples,model.inference.nMinibatch,replace=false)
        model.inference.xview = view.(model.X,[model.inference.MBIndices],:)
        model.inference.yview = view_y.(model.likelihood,model.y,[model.inference.MBIndices])
    end
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    update_A!(model)
    variational_updates!(model);
end

function update_parameters!(model::VStP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    local_prior_updates!(model);
    variational_updates!(model);
end

function computeMatrices!(model::TGP) where {T,TGP<:AbstractGP{T}}
    if model.inference.HyperParametersUpdated
        compute_K!.(model.f,[model.inference.xview],T(jitter))
    end
    model.inference.HyperParametersUpdated=false
end

@traitfn function computeMatrices!(model::TGP) where {T,TGP<:AbstractGP{T};!IsFull{TGP}}
    if model.inference.HyperParametersUpdated
        compute_K!.(model.f,T(jitter))
    end
    #If change of hyperparameters or if stochatic
    if model.inference.HyperParametersUpdated || model.inference.Stochastic
        compute_κ!.(model.f,[model.inference.xview],T(jitter))
    end
    model.inference.HyperParametersUpdated=false
end

function computeMatrices!(model::MOARGP{T}) where {T}
    if model.inference.HyperParametersUpdated
        compute_K!.(model.f,T(jitter))
    end
    #If change of hyperparameters or if stochatic
    if model.inference.HyperParametersUpdated || model.inference.Stochastic
        compute_κ!.(model.f,model.inference.xview,T(jitter))
    end
    model.inference.HyperParametersUpdated=false
end


function computeMatrices!(model::VStP{T,<:Likelihood,<:Inference}) where {T}
    if model.inference.HyperParametersUpdated
        model.Knn .= Symmetric.(KernelModule.kernelmatrix.([model.inference.x],model.kernel) .+ getvariance.(model.kernel).*T(jitter).*[I])
        model.invL .= inv.(getproperty.(cholesky.(model.Knn),:L))
        model.invKnn .= Symmetric.(inv.(cholesky.(model.Knn)))
        # model.invKnn .= Symmetric.(model.invL.*transpose.(model.invL))
    end
end
