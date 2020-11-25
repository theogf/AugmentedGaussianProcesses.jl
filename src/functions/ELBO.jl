"""
    ELBO

ELBO function directly as a function of the variational and hyperparameters
"""
function ELBO(
    model::SVGP{T,L,<:AnalyticVI},
    X,
    Y,
    μ = mean(model[1]),
    Σ = cov(model[1]),
    kernel = kernel(model[1]),
    Z = Zview(model[1]).Z,
    μ₀ = pr_mean(model[1]);
    ρ = getρ(inference(model)),
    likelihood = likelihood(model),
    inference = inference(model),
) where {T<:Real,L<:Likelihood}
    F = zero(T) # Free energy (negative ELBO)
    Kᵤᵤ = kernelmatrix(kernel, Z) + T(jitt) * I
    Kₓᵤ = kernelmatrix(kernel, X, Z)
    Kₓₓ = kerneldiagmatrix(kernel, X) .+ T(jitt)
    K̃ = Kₓₓ - diag_ABt(Kₓᵤ / Kᵤᵤ, Kₓᵤ)
    κ = Kₓᵤ / Kᵤᵤ
    m₀ = μ₀(Z)
    μf = κ * μ
    Σf = diag_ABt(κ * Σ, κ) + K̃
    F -= ρ * expec_log_likelihood(likelihood, inference, Y, μf, Σf)
    F += GaussianKL(μ, m₀, Σ, Kᵤᵤ)
    F += ρ * AugmentedKL(likelihood, Y)
    return -F
end