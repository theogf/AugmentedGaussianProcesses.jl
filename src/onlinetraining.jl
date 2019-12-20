
""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,conv_function=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv_function::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(model::OnlineSVGP,X::AbstractArray,y::AbstractArray;iterations::Integer=2,callback::Union{Nothing,Function}=nothing,Convergence=0)
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
function update_parameters!(model::OnlineSVGP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end


function updateZ!(model::OnlineSVGP)
    for gp in model.f
        add_point!(gp.Z,model.X,model.y,gp.kernel)
    end
end

function compute_local_from_prev!(model::OnlineSVGP{T}) where {T<:Real}
    model.Kmm .= broadcast((Z,kernel)->Symmetric(KernelModule.kernelmatrix(Z,kernel)+getvariance(kernel)*convert(T,Jittering())*I),model.Z,model.kernel)
    model.Knm .= kernelmatrix.([model.X],model.Z,model.kernel)
    model.κ .= model.Knm.*inv.(model.Kmm)
    # local_updates!(model)
end

function save_old_parameters!(model::OnlineSVGP)
    for gp in model.f
        save_old_gp!(gp)
    end
end

function save_old_gp!(gp::_OSVGP{T}) where {T}
    remove_point!(gp.Z,kernelmatrix(gp.kernel, gp.Z),gp.kernel)
    gp.Zₐ = copy(gp.Z.Z)
    gp.invDₐ = Symmetric(-2.0*gp.η₂-inv(gp.K))
    gp.prevη₁ = copy(gp.η₁)
    gp.prev𝓛ₐ = -logdet(gp.Σ) + logdet(gp.K) - dot(gp.μ,gp.η₁)
end

function init_onlinemodel(model::OnlineSVGP{T},X,y) where {T<:Real}
    for gp in model.f
        init_onlinegp!(gp,X,y)
    end
    model.inference.HyperParametersUpdated=false

end

function init_onlinegp!(gp::_OSVGP{T},X,y,jitter::T=T(Jittering())) where {T}
    init!(gp.Z,X,y,gp.kernel)
    nSamples = size(X,1)
    gp.nDim = size(X,2)
    gp.nFeatures = gp.Z.k
    gp.Zₐ = copy(gp.Z.Z)
    gp.μ = zeros(T,gp.nFeatures); gp.η₁ = zero(gp.μ);
    gp.Σ = Symmetric(Matrix(Diagonal(one(T)*I,gp.Z.k)));
    gp.η₂ = -0.5*inv(gp.Σ);
    gp.K = PDMat(first(gp.σ_k)*(kernelmatrix(gp.kernel,gp.Z,obsdim=1)+jitter*I))
    gp.Kab = copy(gp.K.mat)
    gp.κₐ = Diagonal{T}(I,gp.nFeatures)
    gp.K̃ₐ = 2.0*gp.Kab
    gp.Knm = kernelmatrix(gp.kernel,X,gp.Z,obsdim=1)
    gp.κ = gp.Knm/gp.K
    gp.K̃ = first(gp.σ_k)*(kerneldiagmatrix(gp.kernel,X,obsdim=1) .+ jitter) - opt_diag(gp.κ,gp.Knm)
    @assert count(broadcast(x->x.<0,gp.K̃))==0 "K̃ has negative values"
    gp.invDₐ = Symmetric(zeros(T, gp.nFeatures, gp.nFeatures))
    gp.prev𝓛ₐ = zeros(gp.nLatent)
    gp.prevη₁  = zero(gp.η₁)
end
