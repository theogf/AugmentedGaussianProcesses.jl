"""
    train!(model::AbstractGPModel, X::AbstractMatrix, y::AbstractArray; obsdim = 1, iterations::Int=10,callback=nothing,conv=0)
    train!(model::AbstractGPModel, X::AbstractVector, y::AbstractArray; iterations::Int=20,callback=nothing,conv=0)

Function to train the given GP `model`.



## Keyword Arguments

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(
    m::OnlineSVGP,
    X::AbstractMatrix,
    y::AbstractArray,
    state=nothing;
    iterations::Int=20,
    callback=nothing,
    convergence=nothing,
    obsdim::Int=1,
)
    return train!(
        m,
        KernelFunctions.vec_of_vecs(X; obsdim=obsdim),
        y,
        state;
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

    data = wrap_data(X, y) # Set the data in the model
    if is_stochastic(m)
        0 < batchsize(inference(m)) <= n_sample(data) || error(
            "The size of mini-batch $(batchsize(inference(m))) is incorrect (negative or bigger than number of samples), please set `batchsize` correctly in the inference object",
        )
        set_ρ!(model, n_sample(data) / batchsize(inference))
    else
        set_batchsize!(inference(m), n_sample(data))
    end

    if n_iter(m) == 0 # The first time data is seen, initialize all parameters
        init_online_model(m, X)
    else
        state = save_old_parameters!(m, state)
        updateZs!(m, X)
    end
    state = isnothing(state) ? init_state(m) : state

    # model.evol_conv = [] # Array to check on the evolution of convergence
    local_iter = 1
    conv = Inf

    while true # Loop until one condition is matched
        try # Allow for keyboard interruption without losing the model
            if local_iter == 1 # This is needed for dealing with change of sizes
                if n_iter(m) == 0 # First iteration Zₐ is empty
                    state = compute_kernel_matrices(m, state, X, true)
                else
                    state = compute_old_matrices(m, state, X)
                end
                local_vars = local_updates!(
                    state.local_vars,
                    likelihood(m),
                    y,
                    mean_f(m, state.kernel_matrices),
                    var_f(m, state.kernel_matrices),
                )
                # We compute the updates given current data but with
                # the previous inducing points to get directly 
                # the right size for the variational parameters
                ∇E_μs = ∇E_μ(m, y, local_vars)
                ∇E_Σs = ∇E_Σ(m, y, local_vars)
                state = compute_kernel_matrices(m, state, X, true)
                natural_gradient!.(
                    m.f,
                    ∇E_μs,
                    ∇E_Σs,
                    ρ(m),
                    opt(inference(m)),
                    Zviews(m),
                    state.kernel_matrices,
                    state.opt_state,
                )
                state = global_update!(m, state)

            else
                state = update_parameters!(m, state, X, y) # Update all the variational parameters
            end
            set_trained!(m, true)
            if !isnothing(callback)
                callback(m, state, n_iter(m)) # Use a callback method if given by user
            end
            if (n_iter(m) % m.atfrequency == 0) && n_iter(m) >= 3
                state = update_hyperparameters!(m, state, X, y) # Update the hyperparameters
            end
            if verbose(m) > 2 || (verbose(m) > 1 && local_iter % 10 == 0)
                print("Iteration : $(n_iter(m)), ")
                print("ELBO is : $(objective(m, state, y))")
                print("\n")
                println("number of points : $(dim(m[1]))")
            end
            ### Print out informations about the convergence
            local_iter += 1
            m.inference.n_iter += 1
            local_iter <= iterations || break # Check if the number of maximum iterations has been reached
        # (iter < model.nEpochs && conv > model.ϵ) || break; # Check if any condition has been broken
        catch e
            if e isa InterruptException
                @warn "Training interrupted by user at iteration $local_iter"
                break
            else
                rethrow(e)
            end
        end
    end
    if verbose(m) > 0
        println(
            "Training ended after",
            local_iter,
            " iterations. Total number of iterations ",
            n_iter(model),
        )
    end
    state = compute_kernel_matrices(m, state, X, true) #Compute final version of the matrices for prediction
    set_trained!(m, true)
    return m, state
end

# Update all variational parameters of the online sparse 
# variational GP Model
function update_parameters!(model::OnlineSVGP, state, X, y)
    state = compute_kernel_matrices(model, state, X) #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    state = variational_updates(model, state, y)
    return state
end

function updateZs!(m::OnlineSVGP, x)
    for gp in m.f
        gp.Z = InducingPoints.updateZ(gp.Z, gp.Zalg, x; kernel=kernel(gp))
        gp.post.dim = length(Zview(gp))
    end
    setHPupdated!(inference(m), true)
    return nothing
end

function save_old_parameters!(m::OnlineSVGP, state)
    opt_state =
        map(m.f, state.opt_state, state.kernel_matrices) do gp, opt_state, kernel_matrices
            previous_gp = save_old_gp!(gp, opt_state.previous_gp, kernel_matrices.K)
            merge(opt_state, (; previous_gp))
        end
    return merge(state, (; opt_state))
end

function save_old_gp!(gp::OnlineVarLatent{T}, previous_gp, K) where {T}
    gp.Zₐ = deepcopy(gp.Z)
    gp.Z = InducingPoints.remove_point(Random.GLOBAL_RNG, gp.Z, gp.Zalg, Matrix(K))# Matrix(pr_cov(gp)))
    invDₐ = Symmetric(-2.0 * nat2(gp) - inv(K)) # Compute Σ⁻¹ₐ - K⁻¹ₐ
    prevη₁ = nat1(gp)
    prev𝓛ₐ = (-logdet(cov(gp)) + logdet(K) - dot(mean(gp), nat1(gp))) / 2
    return merge(previous_gp, (; invDₐ, prevη₁, prev𝓛ₐ))
end

function init_online_model(m::OnlineSVGP{T}, x) where {T<:Real}
    m.f = ntuple(n_latent(m)) do i
        init_online_gp!(m.f[i], x)
    end
    return setHPupdated!(inference(m), false)
end

function init_online_gp!(gp::OnlineVarLatent{T}, x, jitt::T=T(jitt)) where {T}
    Z = InducingPoints.inducingpoints(gp.Zalg, x; kernel=kernel(gp))
    k = length(Z)
    post = OnlineVarPosterior{T}(k)
    prior = GPPrior(kernel(gp), pr_mean(gp))
    return OnlineVarLatent(
        prior, post, Z, typeof(Z)([]), gp.Zalg, gp.Zupdated, gp.opt, gp.Zopt
    )
end

function compute_old_matrices(m::OnlineSVGP{T}, state, x) where {T}
    kernel_matrices = if haskey(state, :kernel_matrices)
        state.kernel_matrices
    else
        ntuple(x -> (;), n_latent(m))
    end
    kernel_matrices = map(m.f, kernel_matrices) do gp, k_mat
        return compute_old_matrices(gp, k_mat, x, T(jitt))
    end
    return merge(state, (; kernel_matrices))
end

function compute_old_matrices(gp::OnlineVarLatent, state, X::AbstractVector, jitt::Real)
    K = cholesky(kernelmatrix(kernel(gp), gp.Zₐ) + jitt * I)
    Knm = kernelmatrix(kernel(gp), X, gp.Zₐ)
    κ = Knm / K
    K̃ = kernelmatrix_diag(kernel(gp), X) .+ jitt - diag_ABt(κ, Knm)
    all(>(0), K̃) || error("K̃ has negative values")
    return merge(state, (; K, Knm, κ, K̃))
end
