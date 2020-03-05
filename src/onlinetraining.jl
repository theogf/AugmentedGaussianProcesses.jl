
""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,conv_function=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv_function::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(model::OnlineSVGP,X::AbstractArray,y::AbstractArray;iterations::Int=2,callback::Union{Nothing,Function}=nothing,Convergence=0)
    model.X,model.y,nLatent,model.likelihood = check_data!(X,y,model.likelihood)

    @assert nLatent == model.nLatent "Data should always contains the same number of outputs"
    @assert iterations > 0  "Number of iterations should be positive"
    model.inference.nMinibatch = model.inference.nSamples = size(X,1)
    model.inference.MBIndices = collect(1:size(X,1))

    if model.inference.nIter == 1 # The first time data is seen, initialize all parameters
        init_onlinemodel(model,X,y)
        model.likelihood = init_likelihood(model.likelihood,model.inference,model.nLatent,size(X,1),model.nFeatures)
    else
        model.inference.xview = view(X,1:model.inference.nMinibatch,:)
        model.inference.yview = view_y(model.likelihood,y,1:model.inference.nMinibatch)
        save_old_parameters!(model)
        model.likelihood = init_likelihood(model.likelihood,model.inference,model.nLatent,size(X,1),model.nFeatures)
        # compute_local_from_prev!(model)
        updateZ!(model);
    end

    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1; conv = Inf;

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            if local_iter == 1
                # println("BLAH")
                compute_old_matrices!(model)
                local_updates!(model.likelihood,get_y(model),mean_f(model),diag_cov_f(model))
                ∇E_μs = ∇E_μ(model.likelihood,model.inference.vi_opt[1],get_y(model))
                ∇E_Σs = ∇E_Σ(model.likelihood,model.inference.vi_opt[1],get_y(model))
                computeMatrices!(model)
                natural_gradient!.(
                    ∇E_μs, ∇E_Σs,
                    # ∇E_μ(model.likelihood,model.inference.vi_opt[1],get_y(model)),
                    # ∇E_Σ(model.likelihood,model.inference.vi_opt[1],get_y(model)),
                    model.inference,model.inference.vi_opt,get_Z(model),model.f)
                global_update!(model)
            else
                update_parameters!(model) #Update all the variational parameters
            end
            model.Trained = true
            if !isnothing(callback)
                callback(model,model.inference.nIter) #Use a callback method if given by user
            end
            if (model.inference.nIter%model.atfrequency == 0) && model.inference.nIter >= 3
                update_hyperparameters!(model) #Update the hyperparameters
            end
            if model.verbose > 2 || (model.verbose > 1  && local_iter%10==0)
                print("Iteration : $(model.inference.nIter) ")
                print("ELBO is : $(ELBO(model))")
                print("\n")
                # println("kernel lengthscale : $(model.f[1].kernel.transform.s[1])")
                println("number of points : $(model.f[1].dim)")
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
        gp.dim = gp.Z.k
    end
    model.inference.HyperParametersUpdated = true
end

function save_old_parameters!(model::OnlineSVGP)
    for gp in model.f
        save_old_gp!(gp)
    end
end

function save_old_gp!(gp::_OSVGP{T}) where {T}
    gp.Zₐ = copy(gp.Z.Z)
    remove_point!(gp.Z, kernelmatrix(gp.kernel, gp.Z, obsdim=1), gp.kernel)
    gp.invDₐ = Symmetric(-2.0*gp.η₂-inv(gp.K).mat)
    gp.prevη₁ = copy(gp.η₁)
    gp.prev𝓛ₐ = -0.5*logdet(gp.Σ) + 0.5*logdet(gp.K) - 0.5*dot(gp.μ,gp.η₁)
end

function init_onlinemodel(model::OnlineSVGP{T},X,y) where {T<:Real}
    for gp in model.f
        init_online_gp!(gp,X,y)
    end
    model.inference.xview = view(X,1:model.inference.nMinibatch,:)
    model.inference.yview = view_y(model.likelihood,y,1:model.inference.nMinibatch)
    model.inference.ρ = 1.0
    model.inference.HyperParametersUpdated=false
end

function init_online_gp!(gp::_OSVGP{T},X,y,jitt::T=T(jitt)) where {T}
    IPModule.init!(gp.Z,X,y,gp.kernel)
    nSamples = size(X,1)
    gp.dim = gp.Z.k
    gp.Zₐ = copy(gp.Z.Z)
    gp.μ = zeros(T,gp.dim); gp.η₁ = zero(gp.μ);
    gp.Σ = Symmetric(Matrix(Diagonal(one(T)*I,gp.Z.k)));
    gp.η₂ = -0.5*Symmetric(inv(gp.Σ));
    gp.K = PDMat(first(gp.σ_k)*(kernelmatrix(gp.kernel,gp.Z.Z,obsdim=1)+jitt*I))

    gp.Kab = copy(gp.K.mat)
    gp.κₐ = Diagonal{T}(I,gp.dim)
    gp.K̃ₐ = zero(gp.Kab)

    gp.Knm = first(gp.σ_k) * kernelmatrix(gp.kernel, X, gp.Z, obsdim=1)
    gp.κ = gp.Knm / gp.K
    gp.K̃ = first(gp.σ_k) * (kerneldiagmatrix(gp.kernel, X, obsdim=1) .+ jitt) - opt_diag(gp.κ, gp.Knm)
    # @show gp.K̃
    @assert all(gp.K̃ .> 0) "K̃ has negative values"

    gp.invDₐ = Symmetric(Matrix{T}(I,gp.dim, gp.dim))
    gp.prev𝓛ₐ = zero(T)
    gp.prevη₁  = zero(gp.η₁)
end


function compute_old_matrices!(model::OnlineSVGP{T}) where {T}
    for gp in model.f
        compute_old_matrices!(gp,model.inference.xview,T(jitt))
    end
end


function compute_old_matrices!(gp::_OSVGP,X::AbstractMatrix, jitt::Real)
    gp.K = PDMat(first(gp.σ_k)*(kernelmatrix(gp.kernel,gp.Zₐ,obsdim=1)+jitt*I))
    gp.Knm = first(gp.σ_k) * kernelmatrix(gp.kernel, X, gp.Zₐ, obsdim=1)
    gp.κ = gp.Knm / gp.K
    gp.K̃ = first(gp.σ_k) * (kerneldiagmatrix(gp.kernel, X, obsdim=1) .+ jitt) - opt_diag(gp.κ,gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
