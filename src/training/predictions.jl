#File treating all the prediction functions

const pred_nodes, pred_weights =
    gausshermite(100) |> x -> (x[1] .* sqrt2, x[2] ./ sqrtπ)

"""
    predict_f(m::AbstractGP, X_test, cov::Bool=true, diag::Bool=true)

Compute the mean of the predicted latent distribution of `f` on `X_test` for the variational GP `model`

Return also the diagonal variance if `cov=true` and the full covariance if `diag=false`
"""
predict_f

function _predict_f(
    m::GP{T},
    X_test::AbstractVector;
    cov::Bool = true,
    diag::Bool = true,
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
    m::TGP,
    X_test::AbstractVector;
    cov::Bool = true,
    diag::Bool = true,
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
    m::TGP,
    X_test::AbstractVector;
    cov::Bool = true,
    diag::Bool = true,
) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    k_star = kernelmatrix(kernels(m), [X_test], Zviews(m))
    μf = k_star .* (pr_covs(m) .\ means(m))

    μf =
        [[sum(m.A[i][j] .* μf) for j = 1:m.nf_per_task[i]] for i = 1:nOutput(m)]
    if !cov
        return (μf,)
    end
    A = pr_covs(m) .\ ([I] .- covs(m) ./ pr_covs(m))
    if !diag
        k_starstar = (kernelmatrix.(kernels(m), [X_test]) .+ T(jitt) * [I])
        Σf = k_starstar .- k_star .* A .* transpose.(k_star)
        Σf = [
            [sum(m.A[i][j] .^ 2 .* Σf) for j = 1:m.nf_per_task[i]] for i = 1:nOutput(m)
        ]
        return μf, Σf
    else
        k_starstar =
            kernelmatrix_diag.(kernels(m), [X_test]) .+
            [T(jitt) * ones(T, length(X_test))]
        σ²f = k_starstar .- diag_ABt.(k_star .* A, k_star)
        σ²f = [
            [sum(m.A[i][j] .^ 2 .* σ²f) for j = 1:m.nf_per_task[i]] for i = 1:nOutput(m)
        ]
        return μf, σ²f
    end
end

function _predict_f(
    m::MCGP{T},
    X_test::AbstractVector;
    cov::Bool = true,
    diag::Bool = true,
) where {T}
    k_star = kernelmatrix.(kernels(m), [X_test], Zviews(m))
    f = _sample_f(m, X_test, k_star)
    μf = Tuple(vec(StatsBase.mean(f[k], dims = 2)) for k = 1:nLatent(m))
    if !cov
        return (μf,)
    end
    if !diag
        k_starstar = kernelmatrix.(kernels(m), [X_test]) + T(jitt)
        Σf = Symmetric.(k_starstar .- invquad.(pr_covs(m), k_star) .+ StatsBase.cov.(f))
        return μf, Σf
    else
        k_starstar =
            kernelmatrix_diag.(kernels(m), [X_test]) .+
            [T(jitt) * ones(T, length(X_test))]
        σ²f =
            k_starstar .- diag_ABt.(k_star ./ pr_covs(m), k_star) .+
            StatsBase.var.(f, dims = 2)
        return μf, σ²f
    end
end

function _sample_f(
    m::MCGP{T,<:AbstractLikelihood,<:GibbsSampling},
    X_test::AbstractVector,
    k_star = kernelmatrix.(
        kernels(m),
        [X_test],
        Zviews(m),
    ),
) where {T}
    return f = [
        k_star[k] * (pr_cov(m.f[k]) \ inference(m).sample_store[:, :, k]')
        for k = 1:nLatent(m)
    ]
end

## Wrapper for vector input ##
predict_f(
    model::AbstractGP,
    X_test::AbstractVector{<:Real};
    cov::Bool = false,
    diag::Bool = true,
) = predict_f(model, [X_test], cov = cov, diag = diag)

predict_f(
    model::AbstractGP,
    X_test::AbstractMatrix;
    cov::Bool = false,
    diag::Bool = true,
    obsdim::Int = 1,
) = predict_f(
    model,
    KernelFunctions.vec_of_vecs(X_test, obsdim = obsdim),
    cov = cov,
    diag = diag,
)

##
@traitfn predict_f(
    model::TGP,
    X_test::AbstractVector;
    cov::Bool = false,
    diag::Bool = true,
) where {TGP; !IsMultiOutput{TGP}} =
    first.(_predict_f(model, X_test; cov = cov, diag = diag))

@traitfn predict_f(
    model::TGP,
    X_test::AbstractVector;
    cov::Bool = false,
    diag::Bool = true,
) where {TGP; IsMultiOutput{TGP}} =
    _predict_f(model, X_test; cov = cov, diag = diag)


## Wrapper to predict vectors ##
function predict_y(model::AbstractGP, X_test::AbstractVector{<:Real})
    return predict_y(model, [X_test])
end

function predict_y(model::AbstractGP, X_test::AbstractMatrix; obsdim::Int = 1)
    return predict_y(
        model,
        KernelFunctions.vec_of_vecs(X_test, obsdim = obsdim),
    )
end

"""
    predict_y(model::AbstractGP, X_test::AbstractVector)
    predict_y(model::AbstractGP, X_test::AbstractMatrix; obsdim = 1)

Return
    - the predictive mean of `X_test` for regression
    - the sign of `X_test` for classification
    - the most likely class for multi-class classification
    - the expected number of events for an event likelihood
"""
@traitfn predict_y(
    model::TGP,
    X_test::AbstractVector,
) where {TGP <: AbstractGP; !IsMultiOutput{TGP}} =
    predict_y(model.likelihood, first(_predict_f(model, X_test, cov = false)))

@traitfn predict_y(
    model::TGP,
    X_test::AbstractVector,
) where {TGP <: AbstractGP; IsMultiOutput{TGP}} =
    predict_y.(model.likelihood, _predict_f(model, X_test, cov = false))


predict_y(
    l::MultiClassLikelihood,
    μs::AbstractVector{<:AbstractVector{<:Real}},
) = [l.class_mapping[argmax([μ[i] for μ in μs])] for i = 1:length(μs[1])]

predict_y(
    l::MultiClassLikelihood,
    μs::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}},
) = predict_y(l, first(μs))

predict_y(l::EventLikelihood, μ::AbstractVector{<:Real}) = expec_count(l, μ)

predict_y(l::EventLikelihood, μ::AbstractVector{<:AbstractVector}) =
    expec_count(l, first(μ))

## Wrapper to return proba on vectors ##
proba_y(model::AbstractGP, X_test::AbstractVector{<:Real}) =
    proba_y(model, [X_test])

proba_y(model::AbstractGP, X_test::AbstractMatrix; obsdim::Int = 1) =
    proba_y(model, KernelFunctions.vec_of_vecs(X_test, obsdim = obsdim))

"""
    proba_y(model::AbstractGP, X_test::AbstractVector)
    proba_y(model::AbstractGP, X_test::AbstractMatrix; obsdim = 1)

Return the probability distribution p(y_test|model,X_test) :

    - Tuple of vectors of mean and variance for regression
    - Vector of probabilities of y_test = 1 for binary classification
    - Dataframe with columns and probability per class for multi-class classification
"""


@traitfn function proba_y(
    model::TGP,
    X_test::AbstractVector,
) where {TGP <: AbstractGP; !IsMultiOutput{TGP}}
    μ_f, Σ_f = _predict_f(model, X_test, cov = true)
    pred = compute_proba(model.likelihood, μ_f, Σ_f)
end

@traitfn proba_y(
    model::TGP,
    X_test::AbstractVector,
) where {TGP <: AbstractGP; IsMultiOutput{TGP}} = proba_multi_y(model, X_test)

function proba_multi_y(model::AbstractGP, X_test::AbstractVector)
    μ_f, Σ_f = _predict_f(model, X_test, cov = true)
    preds = compute_proba.(likelihood(model), μ_f, Σ_f)
end

compute_proba(
    l::AbstractLikelihood,
    μ::AbstractVector{<:AbstractVector},
    σ²::AbstractVector{<:AbstractVector},
) = compute_proba(l, first(μ), first(σ²))

function StatsBase.mean_and_var(lik, fs)
    @show typeof(fs), size(fs)
    vals = lik.(eachcol(fs))
    @show typeof(vals), size(vals)
    return StatsBase.mean(vals), StatsBase.var(vals)
end

function proba_y(
    model::MCGP{T},
    X_test::AbstractVector;
    nSamples::Int = 200,
) where {T<:Real}
    nTest = length(X_test)
    f = first(_sample_f(model, X_test))
    return proba, sig_proba =  mean_and_var(x->compute_proba.(likelihood(model), x), f)
    # if nLatent(model) == 1
    #     return (proba[1], sig_proba[1])
    # else
    #     return (proba, sig_proba)
    # end
end

#
# function proba_y(model::VGP{T,<:MultiClassLikelihood{T},<:GibbsSampling{T}},X_test::AbstractMatrix{T};nSamples::Int=200) where {T}
#     k_star = kernelmatrix.([X_test],[model.inference.x],model.kernel)
#     f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
#     k_starstar = kerneldiagmatrix.([X_test],model.kernel)
#     K̃ = k_starstar .- diag_ABt.(k_star.*model.invKnn,k_star) .+ [zeros(size(X_test,1)) for i in 1:model.nLatent]
#     nf = length(model.inference.sample_store[1])
#     proba = zeros(size(X_test,1),model.nLatent)
#     labels = Array{Symbol}(undef,model.nLatent)
#     for i in 1:nf
#         res = compute_proba(model.likelihood,getindex.(f,[i]),K̃,nSamples)
#         if i ==  1
#             labels = names(res)
#         end
#         proba .+= Matrix(res)
#     end
#     return DataFrame(proba/nf,labels)
# end
#
function compute_proba(
    l::AbstractLikelihood,
    ::AbstractVector,
    ::AbstractVector,
) where {T<:Real}
    error("Non implemented for the likelihood $l")
end
