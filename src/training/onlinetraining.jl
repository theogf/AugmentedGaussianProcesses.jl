"""
    train!(model::AbstractGPModel, X::AbstractMatrix, y::AbstractVector;obsdim = 1, iterations::Int=10,callback=nothing,conv=0)
    train!(model::AbstractGPModel, X::AbstractVector, y::AbstractVector;iterations::Int=20,callback=nothing,conv=0)

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
    y::AbstractArray,
    state=nothing;
    iterations::Int=20,
    callback::Union{Nothing,Function}=nothing,
    conv::Union{Nothing,Function}=nothing,
)
    iterations > 0 || error("Number of iterations should be positive")
    X, _ = wrap_X(X)
    y = check_data!(y, likelihood(m))

    data = wrap_data!(X, y) # Set the data in the model
        if is_stochastic(model)
        0 < batchsize(inference(model)) <= n_sample(data) || error(
            "The size of mini-batch $(batchsize(inference(model))) is incorrect (negative or bigger than number of samples), please set `batchsize` correctly in the inference object",
        )
        set_ρ!(model, n_sample(data) / batchsize(inference))
    else
        set_batchsize!(inference(model), n_sample(data))
    end

    state = isnothing(state) ? init_state(model) : state
    if n_iter(m) == 1 # The first time data is seen, initialize all parameters
        init_online_model(m, data)
    else
        save_old_parameters!(m)
        updateZ!(m, X)
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
                    ∇E_μs, ∇E_Σs, ρ(m), get_opt(inference(m)), Zviews(m), m.f
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

function InducingPoints.updateZ!(m::OnlineSVGP, x)
    for gp in m.f
        gp.Z = InducingPoints.updateZ(gp.Z, gp.Zalg, x; kernel=kernel(gp))
        gp.post.dim = length(Zview(gp))
    end
    setHPupdated!(inference(m), true)
    return nothing
end

function save_old_parameters!(m::OnlineSVGP, state)
    opt_state = map(m.f, state.opt_state) do gp, opt_state
        previous_gp = save_old_gp!(gp, opt_state.previous_gp)
        merge(opt_state, (; previous_gp))
    end
    merge(state, (; opt_state))
end

function save_old_gp!(gp::OnlineVarLatent{T}, previous_gp) where {T}
    Zₐ = deepcopy(gp.Z)
    # gp.Z = InducingPoints.remove_point(Random.GLOBAL_RNG, gp.Z, gp.Zalg, Matrix(pr_cov(gp)))# Matrix(pr_cov(gp)))
    invDₐ = Symmetric(-2.0 * nat2(gp) - inv(pr_cov(gp))) # Compute Σ⁻¹ₐ - K⁻¹ₐ
    previous_gp.prevη₁ .= nat1(gp)
    prev𝓛ₐ = (-logdet(cov(gp)) + logdet(pr_cov(gp)) - dot(mean(gp), nat1(gp))) / 2
    return merge(previous_gp, (; Zₐ, invDₐ, prev𝓛ₐ))
end

function init_online_model(m::OnlineSVGP{T}, data) where {T<:Real}
    m.f = ntuple(length(m.f)) do i
        init_online_gp!(m.f[i], m, data)
    end
    return setHPupdated!(inference(m), false)
end

function init_online_gp!(gp::OnlineVarLatent{T}, m::OnlineSVGP, jitt::T=T(jitt)) where {T}
    Z = InducingPoints.initZ(gp.Zalg, input(m); kernel=kernel(gp))
    k = length(Z)
    post = OnlineVarPosterior{T}(k)
    prior = GPPrior(kernel(gp), pr_mean(gp))
    return OnlineVarLatent(
        prior,
        post,
        Z,
        gp.Zalg,
        gp.Zupdated,
        gp.opt,
        gp.Zopt,
    )
end

function compute_old_matrices!(m::OnlineSVGP{T}, state, x) where {T}
    kernel_matrices = map(m.f, state.kernel_matrices, state.opt_state) do gp, kernel_matrices, opt_state
        return compute_old_matrices!(gp, kernel_matrices, opt_state.prev_gp, x, T(jitt))
    end
    return merge(state, (; kernel_matrices))
end

function compute_old_matrices!(gp::OnlineVarLatent, state, prev_gp, X::AbstractVector, jitt::Real)
    K = cholesky(kernelmatrix(kernel(gp), prev_gp.Zₐ) + jitt * I)
    Knm = kernelmatrix(kernel(gp), X, prev_gp.Zₐ)
    κ = Knm / K
    K̃ = kernelmatrix_diag(kernel(gp), X) .+ jitt - diag_ABt(κ, Knm)
    all(K̃ .> 0) || error("K̃ has negative values")
    return merge(state, (;K, Knm, κ, K̃))
end
