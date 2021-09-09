"""
    train!(model::AbstractGP, X::AbstractMatrix, y::AbstractVector;obsdim = 1, iterations::Int=10,callback=nothing,conv=0)
    train!(model::AbstractGP, X::AbstractVector, y::AbstractVector;iterations::Int=20,callback=nothing,conv=0)

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(
    m::OnlineSVGP,
    X::AbstractMatrix,
    y::AbstractArray;
    iterations::Int=20,
    callback=nothing,
    convergence=nothing,
    obsdim::Int=1,
)
    return train!(
        m,
        KernelFunctions.vec_of_vecs(X; obsdim=obsdim),
        y;
        iterations=iterations,
        callback=callback,
    )
end

function train!(
    m::OnlineSVGP,
    X::AbstractVector,
    y::AbstractArray;
    iterations::Int=20,
    callback::Union{Nothing,Function}=nothing,
    conv::Union{Nothing,Function}=nothing,
)
    X, _ = wrap_X(X)
    y, _nLatent, m.likelihood = check_data!(y, likelihood(m))

    wrap_data!(data(m), X, y) # Set the data in the model

    _nLatent == nLatent(m) || "Data should always contains the same number of outputs"
    iterations > 0 || "Number of iterations should be positive"
    setnMinibatch!(inference(m), nSamples(data(m)))
    setnSamples!(inference(m), nSamples(data(m)))
    # setMBIndices!(inference(m), collect(1:nMinibatch(inference(m))))

    if nIter(m) == 1 # The first time data is seen, initialize all parameters
        init_onlinemodel(m)
        m.likelihood = init_likelihood(
            likelihood(m), inference(m), nLatent(m), nSamples(data(m))
        )
    else
        save_old_parameters!(m)
        m.likelihood = init_likelihood(likelihood(m), inference(m), nLatent(m), nSamples(m))
        updateZ!(m)
    end

    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter = 1
    conv = Inf

    while true # Loop until one condition is matched
        try # Allow for keyboard interruption without losing the model
            if local_iter == 1
                compute_old_matrices!(m)
                local_updates!(likelihood(m), yview(m), mean_f(m), var_f(m))
                ∇E_μs = ∇E_μ(m)
                ∇E_Σs = ∇E_Σ(m) # They need to be computed before recomputing the matrices
                compute_kernel_matrices!(m)
                natural_gradient!.(
                    ∇E_μs, ∇E_Σs, getρ(inference(m)), get_opt(inference(m)), Zviews(m), m.f
                )
                global_update!(m)
            else
                update_parameters!(m) #Update all the variational parameters
            end
            set_trained!(m, true)
            if !isnothing(callback)
                callback(m, nIter(m)) #Use a callback method if given by user
            end
            if (nIter(m) % m.atfrequency == 0) && nIter(m) >= 3
                update_hyperparameters!(m) #Update the hyperparameters
            end
            if verbose(m) > 2 || (verbose(m) > 1 && local_iter % 10 == 0)
                print("Iteration : $(nIter(m)), ")
                print("ELBO is : $(objective(m))")
                print("\n")
                println("number of points : $(dim(m[1]))")
            end
            ### Print out informations about the convergence
            local_iter += 1
            m.inference.nIter += 1
            (local_iter <= iterations) || break # Verify if the number of maximum iterations has been reached
        # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
        catch e
            if e isa InterruptException
                @warn "Training interrupted by user at iteration $local_iter"
                break
            else
                rethrow(e)
            end
        end
    end
    # if model.verbose > 0
    # println("Training ended after $local_iter iterations. Total number of iterations $(model.inference.nIter)")
    # end
    compute_kernel_matrices!(m) #Compute final version of the matrices for prediction
    return set_trained!(m, true)
end

# Update all variational parameters of the online sparse 
# variational GP Model
function update_parameters!(model::OnlineSVGP)
    compute_kernel_matrices!(model) #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model)
    return nothing
end

function InducingPoints.updateZ!(m::OnlineSVGP)
    for gp in m.f
        gp.Z = InducingPoints.updateZ(gp.Z, gp.Zalg, input(m); kernel=kernel(gp))
        gp.post.dim = length(Zview(gp))
    end
    setHPupdated!(inference(m), true)
    return nothing
end

function save_old_parameters!(m::OnlineSVGP)
    for gp in m.f
        save_old_gp!(gp)
    end
end

function save_old_gp!(gp::OnlineVarLatent{T}) where {T}
    gp.Zₐ = deepcopy(gp.Z)
    gp.Z = InducingPoints.remove_point(Random.GLOBAL_RNG, gp.Z, gp.Zalg, Matrix(pr_cov(gp)))# Matrix(pr_cov(gp)))
    gp.invDₐ = Symmetric(-2.0 * nat2(gp) - inv(pr_cov(gp))) # Compute Σ⁻¹ₐ - K⁻¹ₐ
    gp.prevη₁ = copy(nat1(gp))
    gp.prev𝓛ₐ = (-logdet(cov(gp)) + logdet(pr_cov(gp)) - dot(mean(gp), nat1(gp))) / 2
    return nothing
end

function init_onlinemodel(m::OnlineSVGP{T}) where {T<:Real}
    m.f = ntuple(length(m.f)) do i
        init_online_gp!(m.f[i], m)
    end
    # for gp in m.f
    #     init_online_gp!(gp, m)
    # end
    setρ!(inference(m), one(T))
    return setHPupdated!(inference(m), false)
end

function init_online_gp!(gp::OnlineVarLatent{T}, m::OnlineSVGP, jitt::T=T(jitt)) where {T}
    Z = InducingPoints.initZ(gp.Zalg, input(m); kernel=kernel(gp))
    k = length(Z)
    Zₐ = deepcopy(Z)
    post = OnlineVarPosterior{T}(k)
    prior = GPPrior(
        kernel(gp), pr_mean(gp), cholesky(kernelmatrix(kernel(gp), Z) + jitt * I)
    )

    Kab = zeros(T, k, k)
    κₐ = Matrix{T}(I(k))
    K̃ₐ = zero(Kab)

    Knm = kernelmatrix(kernel(gp), input(m), Z)
    κ = Knm / (kernelmatrix(kernel(gp), Z) + jitt * I)
    K̃ = kernelmatrix_diag(kernel(gp), input(m)) .+ jitt - diag_ABt(κ, Knm)
    all(K̃ .> 0) || error("K̃ has negative values")

    invDₐ = Symmetric(Matrix{T}(I(k)))
    prev𝓛ₐ = zero(T)
    prevη₁ = zeros(T, k)
    return OnlineVarLatent(
        prior,
        post,
        Z,
        gp.Zalg,
        Knm,
        κ,
        K̃,
        gp.Zupdated,
        gp.opt,
        gp.Zopt,
        Zₐ,
        Kab,
        κₐ,
        K̃ₐ,
        invDₐ,
        prev𝓛ₐ,
        prevη₁,
    )
    # return nothing
end

function compute_old_matrices!(m::OnlineSVGP{T}) where {T}
    for gp in m.f
        compute_old_matrices!(gp, xview(m), T(jitt))
    end
end

function compute_old_matrices!(gp::OnlineVarLatent, X::AbstractVector, jitt::Real)
    gp.prior.K = cholesky(kernelmatrix(kernel(gp), gp.Zₐ) + jitt * I)
    gp.Knm = kernelmatrix(kernel(gp), X, gp.Zₐ)
    gp.κ = gp.Knm / pr_cov(gp)
    gp.K̃ = kernelmatrix_diag(kernel(gp), X) .+ jitt - diag_ABt(gp.κ, gp.Knm)
    all(gp.K̃ .> 0) || error("K̃ has negative values")
    return nothing
end
