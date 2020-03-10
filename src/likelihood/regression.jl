abstract type RegressionLikelihood{T<:Real} <: Likelihood{T} end

include("gaussian.jl")
include("studentt.jl")
include("laplace.jl")
include("heteroscedastic.jl")
include("matern.jl")

### Return the labels in a vector of vectors for multiple outputs
function treat_labels!(y::AbstractVector{T},likelihood::L) where {T,L<:RegressionLikelihood}
    @assert T<:Real "For regression target(s) should be real valued"
    return y,1,likelihood
end

predict_y(l::RegressionLikelihood,μ::AbstractVector{<:Real}) = μ
predict_y(l::RegressionLikelihood,μ::AbstractVector{<:AbstractVector}) = first(μ)

function proba_y(model::MCGP{T,<:Likelihood,<:GibbsSampling},X_test::AbstractMatrix{T};nSamples::Int=200) where {T<:Real}
    N_test = size(X_test,1)
    f = _sample_f(model,X_test)
    k_starstar = kerneldiagmatrix.([X_test],model.kernel)
    K̃ = k_starstar .- opt_diag.(k_star.*model.invKnn,k_star) .+ [zeros(size(X_test,1)) for i in 1:model.nLatent]
    nf = length(model.inference.sample_store[1])
    proba = [zeros(size(X_test,1)) for i in 1:model.nLatent]
    sig_proba = [zeros(size(X_test,1)) for i in 1:model.nLatent]
    for i in 1:nf
        for k in 1:model.nLatent
            proba[k], sig_proba[k] = (proba[k],sig_proba[k]) .+ compute_proba(model.likelihood, getindex.(f,[i])[k],K̃[k])
        end
    end
    if model.nLatent == 1
        return (proba[1]/nf, sig_proba[1]/nf)
    else
        return (proba./nf, sig_proba./nf)
    end
end
