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
    iterations::Int = 20,
    callback::Union{Nothing,Function} = nothing,
    conv::Union{Nothing,Function} = nothing,
    obsdim::Int = 1,
    )
    return train!(m, KernelFunctions.vec_of_vecs(X, obsdim=obsdim), y, iterations = iterations)
end


function train!(
    m::OnlineSVGP,
    X::AbstractVector,
    y::AbstractArray;
    iterations::Int = 20,
    callback::Union{Nothing,Function} = nothing,
    conv::Union{Nothing,Function} = nothing,
)
    X, T = wrap_X(X)
    y, _nLatent, m.likelihood = check_data!(y, likelihood(m))

    wrap_data!(data(m), X, y)

    _nLatent == nLatent(m) || "Data should always contains the same number of outputs"
    iterations > 0 || "Number of iterations should be positive"
    setnMinibatch!(inference(m), nSamples(data(m)))
    setnSamples!(inference(m), nSamples(data(m)))
    # setMBIndices!(inference(m), collect(1:nMinibatch(inference(m))))

    if nIter(m) == 1 # The first time data is seen, initialize all parameters
        init_onlinemodel(m)
        m.likelihood = init_likelihood(
            likelihood(m),
            inference(m),
            nLatent(m),
            nSamples(data(m)),
        )
    else
        # setxview!(m.inference, view(X, collect(MBIndices(m), :))
        # setyview!(
            # m.inference,
            # view_y(m.likelihood, y, collect(1:nMinibatch(m.inference))),
        # )
        save_old_parameters!(m)
        m.likelihood = init_likelihood(
            likelihood(m),
            inference(m),
            nLatent(m),
            nSamples(m),
        )
        updateZ!(m)
    end

    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1
    conv = Inf

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            if local_iter == 1
                compute_old_matrices!(m)
                local_updates!(
                    likelihood(m),
                    yview(m),
                    mean_f(m),
                    var_f(m),
                )
                ∇E_μs = ∇E_μ(m)
                ∇E_Σs = ∇E_Σ(m) # They need to be computed before recomputing the matrices
                computeMatrices!(m)
                natural_gradient!.(
                    ∇E_μs,
                    ∇E_Σs,
                    getρ(m.inference),
                    get_opt(m.inference),
                    Zviews(m),
                    m.f,
                )
                global_update!(m)
            else
                update_parameters!(m) #Update all the variational parameters
            end
            set_trained!(m, true)
            if !isnothing(callback)
                callback(m, nIter(m.inference)) #Use a callback method if given by user
            end
            if (nIter(m.inference) % m.atfrequency == 0) &&
               nIter(m.inference) >= 3
                update_hyperparameters!(m) #Update the hyperparameters
            end
            if m.verbose > 2 || (m.verbose > 1 && local_iter % 10 == 0)
                print("Iteration : $(nIter(m.inference)) ")
                print("ELBO is : $(objective(m))")
                print("\n")
                println("number of points : $(m.f[1].dim)")
            end
            ### Print out informations about the convergence
            local_iter += 1
            m.inference.nIter += 1
            (local_iter <= iterations) || break #Verify if the number of maximum iterations has been reached
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
    computeMatrices!(m) #Compute final version of the matrices for prediction
    set_trained!(m, true)
end


"""Update all variational parameters of the online sparse variational GP Model"""
function update_parameters!(model::OnlineSVGP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end


function updateZ!(m::OnlineSVGP)
    for gp in m.f
        InducingPoints.add_point!(gp.Z, m, gp)
        gp.post.dim = length(Zview(gp))
    end
    setHPupdated!(inference(m), true)
end

function save_old_parameters!(m::OnlineSVGP)
    for gp in m.f
        save_old_gp!(gp, m)
    end
end

function save_old_gp!(gp::OnlineVarLatent{T}, m::OnlineSVGP) where {T}
    gp.Zₐ = deepcopy(gp.Z.Z)
    InducingPoints.remove_point!(gp.Z, m, gp)
    gp.invDₐ = Symmetric(-2.0 * nat2(gp) - inv(pr_cov(gp)))
    gp.prevη₁ = copy(nat1(gp))
    gp.prev𝓛ₐ = -0.5*logdet(cov(gp)) + 0.5 * logdet(pr_cov(gp)) - 0.5 * dot(mean(gp), nat1(gp))
end

function init_onlinemodel(m::OnlineSVGP{T}) where {T<:Real}
    for gp in m.f
        init_online_gp!(gp, m)
    end
    setρ!(inference(m), one(T))
    setHPupdated!(inference(m), false)
end

function init_online_gp!(gp::OnlineVarLatent{T}, m::OnlineSVGP, jitt::T = T(jitt)) where {T}
    gp.Z = OptimIP(InducingPoints.init(gp.Z, m, gp), opt(gp.Z))
    k = length(gp.Z)
    gp.Zₐ = vec(gp.Z)
    gp.post = OnlineVarPosterior{T}(k)
    gp.prior = GPPrior(kernel(gp), pr_mean(gp), cholesky(kernelmatrix(kernel(gp), Zview(gp)) + jitt * I))

    gp.Kab = Matrix(pr_cov(gp))
    gp.κₐ = Matrix{T}(I(dim(gp)))
    gp.K̃ₐ = zero(gp.Kab)

    gp.Knm = kernelmatrix(kernel(gp), input(m), gp.Z)
    gp.κ = gp.Knm / pr_cov(gp)
    gp.K̃ =
        kerneldiagmatrix(kernel(gp), input(m)) .+ jitt -
        diag_ABt(gp.κ, gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"

    gp.invDₐ = Symmetric(Matrix{T}(I(dim(gp))))
    gp.prev𝓛ₐ = zero(T)
    gp.prevη₁ = zero(nat1(gp))
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
    gp.K̃ =
        kerneldiagmatrix(kernel(gp), X) .+ jitt -
        diag_ABt(gp.κ, gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
