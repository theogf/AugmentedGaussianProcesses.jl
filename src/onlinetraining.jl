
""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,conv_function=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv_function::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(model::OnlineVGP,X::AbstractArray,y::AbstractArray;iterations::Integer=2,callback::Union{Nothing,Function}=nothing,Convergence=0)
    model.X,model.y,nLatent,model.likelihood = check_data!(X,y,model.likelihood)

    @assert nLatent == model.nLatent "Data should always contains the same number of outputs"
    @assert iterations > 0  "Number of iterations should be positive"

    if model.inference.nIter == 1 # The first time data is seen, initialize all parameters
        init_onlinemodel(model,X,y)
        model.likelihood = init_likelihood(model.likelihood,model.inference,model.nLatent,size(X,1),model.nFeatures)
    else
        save_old_parameters!(model)
        model.likelihood = init_likelihood(model.likelihood,model.inference,model.nLatent,size(X,1),model.nFeatures)
        compute_local_from_prev!(model)
        updateZ!(model);
    end
    model.inference.nSamplesUsed = model.inference.nSamples = size(X,1)
    model.inference.MBIndices = collect(1:size(X,1))

    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1; conv = Inf;

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            setZ!(model)
            if local_iter == 1
                # println("BLAH")
                computeMatrices!(model)
                natural_gradient!(model)
                global_update!(model)
            else
                update_parameters!(model) #Update all the variational parameters
            end
            model.Trained = true
            if !isnothing(callback)
                callback(model,model.inference.nIter) #Use a callback method if put by user
            end
            if !isnothing(model.optimizer) && (model.inference.nIter%model.atfrequency == 0)
                update_hyperparameters!(model) #Update the hyperparameters
            end
            if model.verbose > 2 || (model.verbose > 1  && local_iter%10==0)
                print("Iteration : $(model.inference.nIter) ")
                print("ELBO is : $(ELBO(model))")
                print("\n")
                println("kernel lengthscale : $(getlengthscales(model.kernel[1]))")
            end
            ### Print out informations about the convergence
            local_iter += 1; model.inference.nIter += 1
            (local_iter <= iterations) || break; #Verify if the number of maximum iterations has been reached
            # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
        catch e
            # if isa(e,InterruptException)
                # println("Training interrupted by user at iteration $local_iter");
                # break;
            # else
                rethrow(e)
            # end
        end
    end
    # if model.verbose > 0
      # println("Training ended after $local_iter iterations. Total number of iterations $(model.inference.nIter)")
    # end
    computeMatrices!(model) #Compute final version of the matrices for prediction
    post_process!(model)
    model.Trained = true
end


"""Update all variational parameters of the online sparse variational GP Model"""
function update_parameters!(model::OnlineVGP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end


"""Compute kernel matrices for online variational GPs"""
function computeMatrices!(model::OnlineVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    model.Kmm .= broadcast((Z,kernel)->Symmetric(KernelModule.kernelmatrix(Z,kernel)+getvariance(kernel)*convert(T,Jittering())*I),model.Z,model.kernel)
    model.invKmm .= inv.(model.Kmm)
    model.Kab .= broadcast((Z,Zₐ,kernel)->kernelmatrix(Zₐ,Z,kernel),model.Z,model.Zₐ,model.kernel)
    model.κₐ .= model.Kab.*model.invKmm
    Kₐ = Symmetric.(kernelmatrix.(model.Zₐ,model.kernel)+convert(T,Jittering())*getvariance.(model.kernel).*[I])
    model.K̃ₐ .= Kₐ .+ model.κₐ.*transpose.(model.Kab)
    model.Knm .= kernelmatrix.([model.X],model.Z,model.kernel)
    model.κ .= model.Knm.*model.invKmm
    model.K̃ .= kerneldiagmatrix.([model.X],model.kernel) .+ [convert(T,Jittering())*ones(T,size(model.X,1))] - opt_diag.(model.κ,model.Knm)
    @assert sum(count.(broadcast(x->x.<0,model.K̃)))==0 "K̃ has negative values"
    model.inference.HyperParametersUpdated=false
end


function updateZ!(model::OnlineVGP)
    if !isnothing(model.Zoptimizer)
        add_point!(model.Zalg,model.X,model.y[1],model.kernel[1],optimizer=model.Zoptimizer[1]) #TEMP FOR 1 latent
    else
        add_point!(model.Zalg,model.X,model.y[1],model.kernel[1]) #TEMP FOR 1 latent
    end
end

function compute_local_from_prev!(model::OnlineVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    setZ!(model)
    model.Kmm .= broadcast((Z,kernel)->Symmetric(KernelModule.kernelmatrix(Z,kernel)+getvariance(kernel)*convert(T,Jittering())*I),model.Z,model.kernel)
    model.Knm .= kernelmatrix.([model.X],model.Z,model.kernel)
    model.κ .= model.Knm.*inv.(model.Kmm)
    # local_updates!(model)
end

function setZ!(model::OnlineVGP)
    model.nFeatures = model.Zalg.k
    model.Zupdated = true
    model.Z = fill(model.Zalg.centers,model.nPrior) #TEMP for 1 latent
end

function save_old_parameters!(model::OnlineVGP)
    remove_point!(model.Zalg,kernelmatrix(model.Zalg.centers,model.kernel[1]),model.kernel[1])
    model.Zₐ .= copy.(model.Z)
    model.invDₐ .= Symmetric.(-2.0.*model.η₂.-model.invKmm)
    model.prevη₁ = deepcopy(model.η₁)
    model.prev𝓛ₐ .= -logdet.(model.Σ) + logdet.(model.Kmm) - dot.(model.μ,model.η₁)
end

function init_onlinemodel(model::OnlineVGP{<:Likelihood,<:Inference,T},X,y) where {T<:Real}
    if !isnothing(model.Zoptimizer)
        init!(model.Zalg,X,y[1],model.kernel[1],optimizer=model.Zoptimizer[1])
    else
        init!(model.Zalg,X,y[1],model.kernel[1])
    end
    nSamples = size(X,1)
    model.nDim = size(X,2)
    model.nFeatures = model.Zalg.k
    model.Z = [copy(model.Zalg.centers) for _ in 1:model.nPrior]
    model.Zₐ = copy.(model.Z)
    model.μ = LatentArray([zeros(T,model.nFeatures) for _ in 1:model.nLatent]); model.η₁ = deepcopy(model.μ);
    model.Σ = LatentArray([Symmetric(Matrix(Diagonal(one(T)*I,model.nFeatures))) for _ in 1:model.nLatent]);
    model.η₂ = -0.5*inv.(model.Σ);
    model.μ₀ = [deepcopy(model.μ₀[1]) for _ in 1:model.nPrior]
    model.Kmm = broadcast((Z,kernel)->Symmetric(KernelModule.kernelmatrix(Z,kernel)+getvariance(kernel)*convert(T,Jittering())*I),model.Z,model.kernel)
    model.invKmm = inv.(model.Kmm)
    model.Kab = deepcopy.(model.Kmm)
    model.κₐ = [Diagonal{T}(I,model.nFeatures) for _ in 1:model.nPrior]
    model.K̃ₐ = 2.0.*model.Kab
    model.Knm = kernelmatrix.([model.X],model.Z,model.kernel)
    model.κ = model.Knm.*model.invKmm
    model.K̃ = kerneldiagmatrix.([model.X],model.kernel) .+ [convert(T,Jittering())*ones(T,size(model.X,1))] - opt_diag.(model.κ,model.Knm)
    @assert sum(count.(broadcast(x->x.<0,model.K̃)))==0 "K̃ has negative values"
    model.inference.HyperParametersUpdated=false
    model.invDₐ = LatentArray([Symmetric(zeros(T, model.nFeatures, model.nFeatures)) for _ in 1:model.nPrior])
    model.prev𝓛ₐ  = LatentArray(zeros(model.nLatent))
    model.prevη₁  = zero.(model.η₁)
end
