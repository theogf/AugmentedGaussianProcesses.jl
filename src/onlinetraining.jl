
""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,conv_function=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv_function::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(model::OnlineVGP,X::AbstractArray,y::AbstractArray;iterations::Integer=2,callback=0,Convergence=0)
    model.X,model.y,nLatent,model.likelihood = check_data!(X,y,model.likelihood)
    @assert nLatent == model.nLatent "Data should always contains the same number of outputs"
    @assert iterations > 0  "Number of iterations should be positive"
    if model.inference.nIter == 1
        init_onlinemodel(model,X,y)
    else
        updateZ!(model);
    end
    model.likelihood = init_likelihood(model.likelihood,model.inference,model.nLatent,size(X,1))
    model.inference.nSamplesUsed = model.inference.nSamples = size(X,1)
    model.inference.MBIndices = collect(1:size(X,1))

    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1; conv = Inf;

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            update_parameters!(model) #Update all the variational parameters
            model.Trained = true
            if callback != 0
                callback(model,model.inference.nIter) #Use a callback method if put by user
            end
            if model.Autotuning && (model.inference.nIter%model.atfrequency == 0)
                update_hyperparameters!(model) #Update the hyperparameters
            end
            ### Print out informations about the convergence
            if model.verbose > 2 || (model.verbose > 1  && local_iter%10==0)
                print("Iteration : $(model.inference.nIter) ")
                 print("ELBO is : $(ELBO(model))")
                 print("\n")
                 println("kernel lengthscale : $(getlengthscales(model.kernel[1]))")
             end
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
    Kₐ = kernelmatrix.(model.Zₐ,model.kernel)
    model.K̃ₐ .= Kₐ .+ model.κₐ.*transpose.(model.Kab)
    model.Knm .= kernelmatrix.([model.X],model.Z,model.kernel)
    model.κ .= model.Knm.*model.invKmm
    model.κold .= kernelmatrix.([model.X],model.Zₐ,model.kernel).*inv.(Kₐ)
    model.K̃ .= kerneldiagmatrix.([model.X],model.kernel) .+ [convert(T,Jittering())*ones(T,size(model.X,1))] - opt_diag.(model.κ,model.Knm)
    @assert sum(count.(broadcast(x->x.<0,model.K̃)))==0 "K̃ has negative values"
    model.inference.HyperParametersUpdated=false
end


function updateZ!(model::OnlineVGP)
    model.Zₐ .= copy.(model.Z)
    model.invDₐ .= Symmetric.(-2.0.*model.η₂.-model.invKmm)
    model.prevη₁ = deepcopy(model.η₁)
    model.prev𝓛ₐ .= -logdet.(model.Σ) + logdet.(model.Kmm) - dot.(model.μ,model.η₁)
    update!(model.Zalg,model.X,model.y[1],model.kernel[1]) #TEMP FOR 1 latent
    remove!(model.Zalg,model.Kmm[1],model.kernel[1])
    model.nFeature = model.Zalg.k
    model.Zupdated = true
    model.Z = fill(model.Zalg.centers,model.nPrior) #TEMP for 1 latent
end

function init_onlinemodel(model::OnlineVGP{<:Likelihood,<:Inference,T},X,y) where {T<:Real}
    init!(model.Zalg,X,y[1],model.kernel[1])
    nSamples = size(X,1)
    model.nDim = size(X,2)
    model.nFeature = model.Zalg.k
    model.Z = [copy(model.Zalg.centers) for _ in 1:model.nPrior]
    model.Zₐ = copy.(model.Z)
    model.μ = LatentArray([zeros(T,model.nFeature) for _ in 1:model.nLatent]); model.η₁ = deepcopy(model.μ);
    model.Σ = LatentArray([Symmetric(Matrix(Diagonal(one(T)*I,model.nFeature))) for _ in 1:model.nLatent]);
    model.η₂ = -0.5*inv.(model.Σ);
    model.κ = LatentArray([zeros(T,nSamples, model.nFeature) for _ in 1:model.nPrior])
    model.κold = LatentArray([zeros(T,nSamples, model.nFeature) for _ in 1:model.nPrior])
    model.Knm = deepcopy(model.κ)
    model.K̃ = LatentArray([zeros(T,nSamples) for _ in 1:model.nPrior])
    model.κₐ = LatentArray([zeros(T, model.nFeature, model.nFeature) for _ in 1:model.nPrior])
    model.Kab = deepcopy(model.κₐ)
    model.K̃ₐ = LatentArray([zeros(T, model.nFeature, model.nFeature) for _ in 1:model.nPrior])
    model.invDₐ = LatentArray([Symmetric(zeros(T, model.nFeature, model.nFeature)) for _ in 1:model.nPrior])
    model.prev𝓛ₐ  = LatentArray(zeros(model.nLatent))
    model.prevη₁  = copy.(model.η₁)
    model.Kmm = LatentArray([similar(model.Σ[1]) for _ in 1:model.nPrior]); model.invKmm = similar.(model.Kmm)
end
