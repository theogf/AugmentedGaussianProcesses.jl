#File treating all the prediction functions

const pred_nodes, pred_weights = (x -> (x[1] .* sqrt2, x[2] ./ sqrtπ))(gausshermite(100))

"""
    predict_f(m::AbstractGP, X_test, cov::Bool=true, diag::Bool=true)

Compute the mean of the predicted latent distribution of `f` on `X_test` for the variational GP `model`

Return also the diagonal variance if `cov=true` and the full covariance if `diag=false`
"""
predict_f

function _predict_f(
    m::GP{T}, X_test::AbstractVector; cov::Bool=true, diag::Bool=true
) where {T}
    k_star = kernelmatrix(kernel(m.f), X_test, input(m))
    μf = k_star * mean(m.f) # k * α
    if !cov
        return (μf,)
    end
    if !diag
        k_starstar = kernelmatrix(kernel(m.f), X_test) + T(jitt) * I
        covf = Symmetric(k_starstar - k_star' * (AGP.cov(m.f) \ k_star)) # k** - k* Σ⁻¹ k*
        return μf, covf
    else
        k_starstar = kernelmatrix_diag(kernel(m.f), X_test) .+ T(jitt)
        varf = k_starstar - diag_ABt(k_star / AGP.cov(m.f), k_star)
        return μf, varf
    end
end

@traitfn function _predict_f(
    m::TGP, X_test::AbstractVector; cov::Bool=true, diag::Bool=true
) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    k_star = kernelmatrix.(kernels(m), [X_test], Zviews(m))
    μf = k_star .* (pr_covs(m) .\ means(m))
    if !cov
        return (μf,)
    end
    A = pr_covs(m) .\ (Ref(I) .- covs(m) ./ pr_covs(m))
    if !diag
        k_starstar = kernelmatrix.(kernels(m), Ref(X_test)) .+ T(jitt) * [I]
        Σf = Symmetric.(k_starstar .- k_star .* A .* transpose.(k_star))
        return μf, Σf
    else
        k_starstar =
            kernelmatrix_diag.(kernels(m), Ref(X_test)) .+
            Ref(T(jitt) * ones(T, size(X_test, 1)))
        σ²f = k_starstar .- diag_ABt.(k_star .* A, k_star)
        return μf, σ²f
    end
end

@traitfn function _predict_f(
    m::TGP, X_test::AbstractVector; cov::Bool=true, diag::Bool=true
) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    k_star = kernelmatrix(kernels(m), [X_test], Zviews(m))
    μf = k_star .* (pr_covs(m) .\ means(m))

    μf = [[sum(m.A[i][j] .* μf) for j in 1:m.nf_per_task[i]] for i in 1:nOutput(m)]
    if !cov
        return (μf,)
    end
    A = pr_covs(m) .\ ([I] .- covs(m) ./ pr_covs(m))
    if !diag
        k_starstar = (kernelmatrix.(kernels(m), [X_test]) .+ T(jitt) * [I])
        Σf = k_starstar .- k_star .* A .* transpose.(k_star)
        Σf = [[sum(m.A[i][j] .^ 2 .* Σf) for j in 1:m.nf_per_task[i]] for i in 1:nOutput(m)]
        return μf, Σf
    else
        k_starstar =
            kernelmatrix_diag.(kernels(m), [X_test]) .+ [T(jitt) * ones(T, length(X_test))]
        σ²f = k_starstar .- diag_ABt.(k_star .* A, k_star)
        σ²f = [
            [sum(m.A[i][j] .^ 2 .* σ²f) for j in 1:m.nf_per_task[i]] for i in 1:nOutput(m)
        ]
        return μf, σ²f
    end
end

function _predict_f(
    m::MCGP{T}, X_test::AbstractVector; cov::Bool=true, diag::Bool=true
) where {T}
    k_star = kernelmatrix.(kernels(m), [X_test], Zviews(m))
    f = _sample_f(m, X_test, k_star)
    μf = mean.(f)
    if !cov
        return (μf,)
    end
    if !diag
        k_starstar = kernelmatrix.(kernels(m), [X_test]) + T(jitt)
        Σf = Symmetric.(k_starstar .- invquad.(pr_covs(m), k_star) .+ StatsBase.cov.(f))
        return μf, Σf
    else
        k_starstar =
            kernelmatrix_diag.(kernels(m), [X_test]) .+ [T(jitt) * ones(T, length(X_test))]
        σ²f = k_starstar .- diag_ABt.(k_star ./ pr_covs(m), k_star) .+ StatsBase.var.(f)
        return μf, σ²f
    end
end

function _sample_f(
    m::MCGP{T,<:AbstractLikelihood,<:GibbsSampling},
    X_test::AbstractVector,
    k_star=kernelmatrix.(kernels(m), [X_test], Zviews(m)),
) where {T}
    return f = [
        Ref(k_star[k] / pr_cov(m.f[k])) .* getindex.(inference(m).sample_store, k) for
        k in 1:nLatent(m)
    ]
end

## Wrapper for vector input ##
function predict_f(
    model::AbstractGP, X_test::AbstractVector{<:Real}; cov::Bool=false, diag::Bool=true
)
    return predict_f(model, reshape(X_test, :, 1); cov=cov, diag=diag)
end

function predict_f(
    model::AbstractGP,
    X_test::AbstractMatrix;
    cov::Bool=false,
    diag::Bool=true,
    obsdim::Int=1,
)
    return predict_f(
        model, KernelFunctions.vec_of_vecs(X_test; obsdim=obsdim); cov=cov, diag=diag
    )
end

##
@traitfn function predict_f(
    model::TGP, X_test::AbstractVector; cov::Bool=false, diag::Bool=true
) where {TGP; !IsMultiOutput{TGP}}
    return first.(_predict_f(model, X_test; cov=cov, diag=diag))
end

@traitfn function predict_f(
    model::TGP, X_test::AbstractVector; cov::Bool=false, diag::Bool=true
) where {TGP; IsMultiOutput{TGP}}
    return _predict_f(model, X_test; cov=cov, diag=diag)
end

## Wrapper to predict vectors ##
function predict_y(model::AbstractGP, X_test::AbstractVector{<:Real})
    return predict_y(model, reshape(X_test, :, 1))
end

function predict_y(model::AbstractGP, X_test::AbstractMatrix; obsdim::Int=1)
    return predict_y(model, KernelFunctions.vec_of_vecs(X_test; obsdim=obsdim))
end

"""
    predict_y(model::AbstractGP, X_test::AbstractVector)
    predict_y(model::AbstractGP, X_test::AbstractMatrix; obsdim = 1)

Return
    - the predictive mean of `X_test` for regression
    - 0 or 1 of `X_test` for classification
    - the most likely class for multi-class classification
    - the expected number of events for an event likelihood
"""
@traitfn function predict_y(
    model::TGP, X_test::AbstractVector
) where {TGP <: AbstractGP; !IsMultiOutput{TGP}}
    return predict_y(likelihood(model), first(_predict_f(model, X_test; cov=false)))
end

@traitfn function predict_y(
    model::TGP, X_test::AbstractVector
) where {TGP <: AbstractGP; IsMultiOutput{TGP}}
    return predict_y.(likelihood(model), _predict_f(model, X_test; cov=false))
end

function predict_y(l::MultiClassLikelihood, μs::AbstractVector{<:AbstractVector{<:Real}})
    return [l.class_mapping[argmax([μ[i] for μ in μs])] for i in 1:length(μs[1])]
end

function predict_y(
    l::MultiClassLikelihood, μs::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}}
)
    return predict_y(l, first(μs))
end

predict_y(l::EventLikelihood, μ::AbstractVector{<:Real}) = expec_count(l, μ)

function predict_y(l::EventLikelihood, μ::AbstractVector{<:AbstractVector})
    return expec_count(l, first(μ))
end

## Wrapper to return proba on vectors ##
function proba_y(model::AbstractGP, X_test::AbstractVector{<:Real})
    return proba_y(model, reshape(X_test, :, 1))
end

function proba_y(model::AbstractGP, X_test::AbstractMatrix; obsdim::Int=1)
    return proba_y(model, KernelFunctions.vec_of_vecs(X_test; obsdim=obsdim))
end

"""
    proba_y(model::AbstractGP, X_test::AbstractVector)
    proba_y(model::AbstractGP, X_test::AbstractMatrix; obsdim = 1)

Return the probability distribution p(y_test|model,X_test) :

    - Tuple of vectors of mean and variance for regression
    - Vector of probabilities of y_test = 1 for binary classification
    - Dataframe with columns and probability per class for multi-class classification
"""

@traitfn function proba_y(
    model::TGP, X_test::AbstractVector
) where {TGP <: AbstractGP; !IsMultiOutput{TGP}}
    μ_f, Σ_f = _predict_f(model, X_test; cov=true)
    return pred = compute_proba(model.likelihood, μ_f, Σ_f)
end

@traitfn function proba_y(
    model::TGP, X_test::AbstractVector
) where {TGP <: AbstractGP; IsMultiOutput{TGP}}
    return proba_multi_y(model, X_test)
end

function proba_multi_y(model::AbstractGP, X_test::AbstractVector)
    μ_f, Σ_f = _predict_f(model, X_test; cov=true)
    return preds = compute_proba.(likelihood(model), μ_f, Σ_f)
end

function compute_proba(
    l::AbstractLikelihood,
    μ::AbstractVector{<:AbstractVector},
    σ²::AbstractVector{<:AbstractVector},
)
    return compute_proba(l, first(μ), first(σ²))
end

function StatsBase.mean_and_var(lik::AbstractLikelihood, fs::AbstractMatrix)
    vals = lik.(eachcol(fs))
    return StatsBase.mean(vals), StatsBase.var(vals)
end

function proba_y(model::MCGP, X_test::AbstractVector{<:Real}; nSamples::Int=100)
    return proba_y(model, reshape(X_test, :, 1); nSamples)
end

function proba_y(model::MCGP, X_test::AbstractMatrix; nSamples::Int=100, obsdim=1)
    return proba_y(model, KernelFunctions.vec_of_vecs(X_test; obsdim); nSamples)
end

function proba_y(model::MCGP, X_test::AbstractVector; nSamples::Int=200)
    nTest = length(X_test)
    f = first(_sample_f(model, X_test))
    return proba, sig_proba = mean_and_var(compute_proba_f.(likelihood(model), f))
end

function compute_proba_f(l::AbstractLikelihood, f::AbstractVector{<:Real})
    return compute_proba.(l, f)
end

function compute_proba(
    l::AbstractLikelihood, ::AbstractVector, ::AbstractVector
) where {T<:Real}
    return error("Non implemented for the likelihood $l")
end

for f in [:predict_f, :predict_y, :proba_y]
end
