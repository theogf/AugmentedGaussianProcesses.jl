"""
    ELBO

ELBO function directly as a function of the variational and hyperparameters
"""
function ELBO(
    gp::SparseVarLatent{T},
    l::Likelihood,
    i::AnalyticVI,
    X,
    Y,
    kernel = kernel(gp),
    μ₀ = pr_mean(model[1]),
    Z = Zview(gp).Z;
    μ = mean(gp),
    Σ = cov(gp),
    ρ = getρ(i),
) where {T<:Real}
    F = zero(T) # Free energy (negative ELBO)
    Kᵤᵤ = kernelmatrix(kernel, Z) + T(jitt) * I
    Kₓᵤ = kernelmatrix(kernel, X, Z)
    Kₓₓ = kerneldiagmatrix(kernel, X) .+ T(jitt)
    K̃ = Kₓₓ - diag_ABt(Kₓᵤ / Kᵤᵤ, Kₓᵤ)
    κ = Kₓᵤ / Kᵤᵤ
    m₀ = μ₀(Z)
    μf = κ * μ
    Σf = diag_ABt(κ * Σ, κ) + K̃
    F -= ρ * expec_log_likelihood(l, i, Y, μf, Σf)
    F += GaussianKL(μ, m₀, Σ, Kᵤᵤ)
    F += ρ * AugmentedKL(l, Y)
    return -F
end

function ELBO(
    model::SVGP{T},
    pr_means,
    kernels,
    Zs,
) where {T<:Real}
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    setZs!(model, Zs)
    computeMatrices!(model, true)
    # model.f[1].Knm = kernelmatrix(kernel(model[1]), xview(model), model.f[1].Z)
    # model.f[1].κ = copy((model[1].prior.K \ model[1].Knm')')
    # model.f[1].K̃ = kerneldiagmatrix(kernel(model.f[1]), xview(model)) .+ T(jitt) -
        # diag_ABt(model[1].κ, model[1].Knm)
    # compute_κ!.(getf(model), [xview(model)], T(jitt))
    return ELBO(model)
end