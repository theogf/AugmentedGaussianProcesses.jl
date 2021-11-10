# File treating all the prediction functions

## Nodes and weights for predictions based on quadrature
const pred_nodes, pred_weights = (x -> (x[1] .* sqrt2, x[2] ./ sqrtπ))(gausshermite(100))

function _predict_f(
    m::GP{T}, X_test::AbstractVector, state=nothing; cov::Bool=true, diag::Bool=true
) where {T}
    k_star = kernelmatrix(kernel(m.f), X_test, input(m.data))
    μf = k_star * mean(m.f) # k * α
    if !cov
        return (μf,)
    end
    if diag
        k_starstar = kernelmatrix_diag(kernel(m.f), X_test) .+ T(jitt)
        varf = k_starstar - diag_ABt(k_star / AGP.cov(m.f), k_star)
        return μf, varf
    else
        k_starstar = kernelmatrix(kernel(m.f), X_test) + T(jitt) * I
        covf = Symmetric(k_starstar - k_star' * (AGP.cov(m.f) \ k_star)) # k** - k* Σ⁻¹ k*
        return μf, covf
    end
end

@traitfn function _predict_f(
    m::TGP, X_test::AbstractVector, state=nothing; cov::Bool=true, diag::Bool=true
) where {T,TGP<:AbstractGPModel{T};!IsMultiOutput{TGP}}
    Ks = if isnothing(state)
        compute_K.(m.f, Zviews(m), T(jitt))
    else
        getproperty.(state.kernel_matrices, :K)
    end
    k_star = kernelmatrix.(kernels(m), (X_test,), Zviews(m))
    μf = k_star .* (Ks .\ means(m))
    if !cov
        return (μf,)
    end
    A = Ks .\ (Ref(I) .- covs(m) ./ Ks)
    if diag
        k_starstar =
            kernelmatrix_diag.(kernels(m), Ref(X_test)) .+
            Ref(T(jitt) * ones(T, size(X_test, 1)))
        σ²f = k_starstar .- diag_ABt.(k_star .* A, k_star)
        return (μf, σ²f)
    else
        k_starstar = kernelmatrix.(kernels(m), Ref(X_test)) .+ T(jitt) * [I]
        Σf = Symmetric.(k_starstar .- k_star .* A .* transpose.(k_star))
        return (μf, Σf)
    end
end

@traitfn function _predict_f(
    m::TGP, X_test::AbstractVector, state=nothing; cov::Bool=true, diag::Bool=true
) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    Ks = if isnothing(state)
        compute_K.(m.f, Zviews(m), T(jitt))
    else
        getproperty.(state.kernel_matrices, :K)
    end
    k_star = kernelmatrix.(kernels(m), (X_test,), Zviews(m))
    μf = k_star .* (Ks .\ means(m))

    μf = ntuple(n_output(m)) do i
        ntuple(m.nf_per_task[i]) do j
            sum(m.A[i][j] .* μf)
        end
    end
    if !cov
        return (μf,)
    end
    A = Ks .\ ([I] .- covs(m) ./ Ks)
    if diag
        k_starstar =
            kernelmatrix_diag.(kernels(m), [X_test]) .+ [T(jitt) * ones(T, length(X_test))]
        σ²f = k_starstar .- diag_ABt.(k_star .* A, k_star)
        σ²f = ntuple(n_output(m)) do i
            ntuple(m.nf_per_task[i]) do j
                sum(m.A[i][j] .^ 2 .* σ²f)
            end
        end
        return μf, σ²f
    else
        k_starstar = (kernelmatrix.(kernels(m), [X_test]) .+ T(jitt) * [I])
        Σf = k_starstar .- k_star .* A .* transpose.(k_star)
        Σf = ntuple(n_output(m)) do i
            ntuple(m.nf_per_task[i]) do j
                sum(m.A[i][j] .^ 2 .* Σf)
            end
        end
        return μf, Σf
    end
end

function _predict_f(
    m::MCGP{T}, X_test::AbstractVector, state=nothing; cov::Bool=true, diag::Bool=true
) where {T}
    Ks = if isnothing(state)
        compute_K.(m.f, Zviews(model), T(jitt))
    else
        getproperty.(state.kernel_matrices, :K)
    end
    k_star = kernelmatrix.(kernels(m), [X_test], Zviews(m))
    f = _sample_f(m, X_test, Ks, k_star)
    μf = mean.(f)
    if !cov
        return (μf,)
    end
    if !diag
        k_starstar = kernelmatrix.(kernels(m), [X_test]) + T(jitt)
        Σf = Symmetric.(k_starstar .- invquad.(Ks, k_star) .+ StatsBase.cov.(f))
        return μf, Σf
    else
        k_starstar =
            kernelmatrix_diag.(kernels(m), [X_test]) .+ [T(jitt) * ones(T, length(X_test))]
        σ²f = k_starstar .- diag_ABt.(k_star ./ Ks, k_star) .+ StatsBase.var.(f)
        return μf, σ²f
    end
end

function _sample_f(
    m::MCGP{T,<:AbstractLikelihood,<:GibbsSampling},
    X_test::AbstractVector,
    Ks=kernelmatrix.(kernels(m), Zviews(m)),
    k_star=kernelmatrix.(kernels(m), [X_test], Zviews(m)),
) where {T}
    return [
        Ref(k_star[k] / Ks[k]) .* getindex.(inference(m).sample_store, k) for
        k in 1:n_latent(m)
    ]
end

"""
    predict_f(m::AbstractGPModel, X_test, cov::Bool=true, diag::Bool=true)

Compute the mean of the predicted latent distribution of `f` on `X_test` for the variational GP `model`

Return also the diagonal variance if `cov=true` and the full covariance if `diag=false`
"""
predict_f

function predict_f(
    model::AbstractGPModel,
    X_test::AbstractMatrix,
    state=nothing;
    cov::Bool=false,
    diag::Bool=true,
    obsdim::Int=1,
)
    return predict_f(
        model, KernelFunctions.vec_of_vecs(X_test; obsdim=obsdim), state; cov=cov, diag=diag
    )
end

function predict_f(
    model::AbstractGPModel, X_test::AbstractVector, state=nothing; cov::Bool=false, diag::Bool=true
)
    if n_latent(model) > 1
        return _predict_f(model, X_test, state; cov=cov, diag=diag)
    else
        return only.(_predict_f(model, X_test, state; cov=cov, diag=diag))
    end
end

"""
    predict_y(model::AbstractGPModel, X_test::AbstractVector)
    predict_y(model::AbstractGPModel, X_test::AbstractMatrix; obsdim = 1)

Return
    - the predictive mean of `X_test` for regression
    - 0 or 1 of `X_test` for classification
    - the most likely class for multi-class classification
    - the expected number of events for an event likelihood
"""
predict_y

function predict_y(
    model::AbstractGPModel, X_test::AbstractMatrix, state=nothing; obsdim::Int=1
)
    return predict_y(model, KernelFunctions.vec_of_vecs(X_test; obsdim=obsdim), state)
end

@traitfn function predict_y(
    model::TGP, X_test::AbstractVector, state=nothing
) where {TGP <: AbstractGPModel; !IsMultiOutput{TGP}}
    return predict_y(likelihood(model), only(_predict_f(model, X_test, state; cov=false)))
end

@traitfn function predict_y(
    model::TGP, X_test::AbstractVector, state=nothing
) where {TGP <: AbstractGPModel; IsMultiOutput{TGP}}
    return predict_y.(
        likelihood(model), only.(_predict_f(model, X_test, state; cov=false))
    )
end

function predict_y(l::MultiClassLikelihood, μs::Tuple{Vararg{<:AbstractVector{<:Real}}})
    return [l.class_mapping[argmax([μ[i] for μ in μs])] for i in 1:length(μs[1])]
end

function predict_y(
    l::MultiClassLikelihood,
    μs::Tuple{<:Tuple{Vararg{<:AbstractVector{<:AbstractVector{<:Real}}}}},
)
    return predict_y(l, only(μs))
end

predict_y(l::EventLikelihood, μ::AbstractVector{<:Real}) = mean.(l.(μ))

function predict_y(l::EventLikelihood, μ::Tuple{<:AbstractVector})
    return predict_y(l, only(μ))
end

"""
    proba_y(model::AbstractGPModel, X_test::AbstractVector)
    proba_y(model::AbstractGPModel, X_test::AbstractMatrix; obsdim = 1)

Return the probability distribution p(y_test|model,X_test) :

    - Tuple of vectors of mean and variance for regression
    - Vector of probabilities of y_test = 1 for binary classification
    - Dataframe with columns and probability per class for multi-class classification
"""
proba_y

function proba_y(
    model::AbstractGPModel, X_test::AbstractMatrix, state=nothing; obsdim::Int=1
)
    return proba_y(model, KernelFunctions.vec_of_vecs(X_test; obsdim=obsdim))
end

@traitfn function proba_y(
    model::TGP, X_test::AbstractVector, state=nothing
) where {TGP <: AbstractGPModel; !IsMultiOutput{TGP}}
    μ_f, Σ_f = _predict_f(model, X_test, state; cov=true)
    return compute_proba(model.likelihood, μ_f, Σ_f)
end

@traitfn function proba_y(
    model::TGP, X_test::AbstractVector, state=nothing
) where {TGP <: AbstractGPModel; IsMultiOutput{TGP}}
    return proba_multi_y(model, X_test, state)
end

function proba_multi_y(model::AbstractGPModel, X_test::AbstractVector, state)
    μ_f, Σ_f = _predict_f(model, X_test, state; cov=true)
    return preds = compute_proba.(likelihood(model), μ_f, Σ_f)
end

function compute_proba(
    l::AbstractLikelihood, μ::Tuple{<:AbstractVector}, σ²::Tuple{<:AbstractVector}
)
    return compute_proba(l, only(μ), only(σ²))
end

function StatsBase.mean_and_var(lik::AbstractLikelihood, fs::AbstractMatrix)
    vals = lik.(eachcol(fs))
    return StatsBase.mean(vals), StatsBase.var(vals)
end

function proba_y(
    model::MCGP, X_test::AbstractMatrix, state=nothing; nSamples::Int=100, obsdim=1
)
    return proba_y(model, KernelFunctions.vec_of_vecs(X_test; obsdim), state; nSamples)
end

function proba_y(
    model::MCGP{T}, X_test::AbstractVector, state=nothing; nSamples::Int=200
) where {T}
    Ks = if isnothing(state)
        cholesky.(kernelmatrix.(kernels(model), Zviews(model)) .+ Ref(T(jitt) * I))
    else
        getproperty.(state.kernel_matrices, :K)
    end
    f = only(_sample_f(model, X_test, Ks))
    return mean_and_var(compute_proba_f.(likelihood(model), f))
end

function compute_proba_f(l::AbstractLikelihood, f::AbstractVector{<:Real})
    return compute_proba.(l, f)
end

function compute_proba(l::AbstractLikelihood, ::Any, ::Any) where {T<:Real}
    return error("Non implemented for the likelihood $l")
end
