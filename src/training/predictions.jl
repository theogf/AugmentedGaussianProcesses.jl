#File treating all the prediction functions

const pred_nodes, pred_weights =
    gausshermite(100) |> x -> (x[1] .* sqrt2, x[2] ./ sqrtπ)

"""
Compute the mean of the predicted latent distribution of `f` on `X_test` for the variational GP `model`

Return also the diagonal variance if `cov=true` and the full covariance if `diag=false`
"""
predict_f

@traitfn function _predict_f(
    model::TGP,
    X_test::AbstractMatrix{<:Real};
    cov::Bool = true,
    diag::Bool = true,
) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    k_star =
        kernelmatrix.(get_kernel(model), [X_test], get_Z(model), obsdim = 1)
    μf = k_star .* (get_K(model) .\ get_μ(model))
    if !cov
        return (μf,)
    end
    A = get_K(model) .\ (Ref(I) .- get_Σ(model) ./ get_K(model))
    if !diag
        k_starstar =
            kernelmatrix.(get_kernel(model), [X_test], obsdim = 1) .+
            T(jitt) * [I]
        Σf = Symmetric.(k_starstar .- k_star .* A .* transpose.(k_star))
        return μf, Σf
    else
        k_starstar =
            kerneldiagmatrix.(get_kernel(model), [X_test], obsdim = 1) .+
            [T(jitt) * ones(T, size(X_test, 1))]
        σ²f = k_starstar .- opt_diag.(k_star .* A, k_star)
        return μf, σ²f
    end
end

function _predict_f(
    model::GP{T},
    X_test::AbstractMatrix{<:Real};
    cov::Bool = true,
    diag::Bool = true,
) where {T}
    k_star =
        kernelmatrix.(get_kernel(model), [X_test], get_Z(model), obsdim = 1)
    μf = k_star .* mean_f(model)
    if !cov
        return (μf,)
    end
    A = [inv(model.f[1].K).mat]
    if !diag
        k_starstar =
            kernelmatrix.(get_kernel(model), [X_test], obsdim = 1) .+
            T(jitt) * [I]
        Σf = Symmetric.(k_starstar .- k_star .* A .* transpose.(k_star))
        return μf, Σf
    else
        k_starstar =
            kerneldiagmatrix.(get_kernel(model), [X_test], obsdim = 1) .+
            [T(jitt) * ones(T, size(X_test, 1))]
        σ²f = k_starstar .- opt_diag.(k_star .* A, k_star)
        return μf, σ²f
    end
end

@traitfn function _predict_f(
    m::TGP,
    X_test::AbstractVector{<:AbstractMatrix{<:Real}};
    cov::Bool = true,
    diag::Bool = true,
) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    k_star = kernelmatrix.(get_kernel(m), X_test, get_Z(m), obsdim = 1)
    μf = k_star .* (get_K(m) .\ get_μ(m))

    μf = [[sum(m.A[i][j] .* μf) for j = 1:m.nf_per_task[i]] for i = 1:m.nTask]
    if !cov
        return (μf,)
    end
    A = get_K(m) .\ ([I] .- get_Σ(m) ./ get_K(m))
    if !diag
        k_starstar =
            (kernelmatrix.(get_kernel(m), X_test, obsdim = 1) .+ T(jitt) * [I])
        Σf = k_starstar .- k_star .* A .* transpose.(k_star)
        Σf = [
            [sum(m.A[i][j] .^ 2 .* Σf) for j = 1:m.nf_per_task[i]] for i = 1:m.nTask
        ]
        return μf, Σf
    else
        k_starstar =
            kerneldiagmatrix.(get_kernel(m), X_test, obsdim = 1) .+
            T(jitt) .* ones.(T, size.(X_test, 1))
        σ²f = k_starstar .- opt_diag.(k_star .* A, k_star)
        σ²f = [
            [sum(m.A[i][j] .^ 2 .* σ²f) for j = 1:m.nf_per_task[i]] for i = 1:m.nTask
        ]
        return μf, σ²f
    end
end

function _predict_f(
    model::MCGP{T},
    X_test::AbstractMatrix{<:Real};
    cov::Bool = true,
    diag::Bool = true,
) where {T}
    k_star =
        kernelmatrix.(get_kernel(model), [X_test], get_Z(model), obsdim = 1)
    f = _sample_f(model, X_test, k_star)
    μf = Tuple(vec(mean(f[k], dims = 2)) for k = 1:model.nLatent)
    if !cov
        return (μf,)
    end
    if !diag
        k_starstar =
            kernelmatrix.(get_kernel(model), [X_test], obsdim = 1) + T(jitt)
        Σf = Symmetric.(k_starstar .- invquad.(get_K(model), k_star) .+ cov.(f))
        return μf, Σf
    else
        k_starstar =
            kerneldiagmatrix.(get_kernel(model), [X_test], obsdim = 1) .+
            [T(jitt) * ones(T, size(X_test, 1))]
        σ²f =
            k_starstar .- opt_diag.(k_star ./ get_K(model), k_star) .+
            diag.(cov.(f, dims = 2))
        return μf, σ²f
    end
end

function _sample_f(
    model::MCGP{T,<:Likelihood,<:GibbsSampling},
    X_test::AbstractMatrix{T},
    k_star = kernelmatrix.(
        get_kernel(model),
        [X_test],
        get_Z(model),
        obsdim = 1,
    ),
) where {T}
    return f = [
        k_star[k] * (model.f[k].K \ model.inference.sample_store[:, :, k]')
        for k = 1:model.nLatent
    ]
end

## Wrapper for vector input ##
predict_f(
    model::AbstractGP,
    X_test::AbstractVector{T};
    cov::Bool = false,
    diag::Bool = true,
) where {T<:Real} =
    predict_f(model, reshape(X_test, :, 1), cov = cov, diag = diag)

##
@traitfn predict_f(
    model::TGP,
    X_test::AbstractMatrix{<:Real};
    cov::Bool = false,
    diag::Bool = true,
) where {TGP;!IsMultiOutput{TGP}} =
    first.(_predict_f(model, X_test; cov = cov, diag = diag))

@traitfn predict_f(
    model::TGP,
    X_test::AbstractMatrix{<:Real};
    cov::Bool = false,
    diag::Bool = true,
) where {TGP; IsMultiOutput{TGP}} =
    predict_f(model, [X_test]; cov = cov, diag = diag)

@traitfn predict_f(
    model::TGP,
    X_test::AbstractVector{<:AbstractMatrix{<:Real}};
    cov::Bool = false,
    diag::Bool = true,
) where {TGP; IsMultiOutput{TGP}} =
    _predict_f(model, X_test; cov = cov, diag = diag)


## Wrapper to predict vectors ##
function predict_y(model::AbstractGP{T}, X_test::AbstractVector{T}) where {T}
    return predict_y(model, reshape(X_test, :, 1))
end

"""
`predict_y(model::AbstractGP,X_test::AbstractMatrix)`

Return
    - the predictive mean of `X_test` for regression
    - the sign of `X_test` for classification
    - the most likely class for multi-class classification
    - the expected number of events for an event likelihood
"""
@traitfn predict_y(
    model::TGP,
    X_test::AbstractMatrix,
) where {TGP<:AbstractGP; !IsMultiOutput{TGP}} =
    predict_y(model.likelihood, first(_predict_f(model, X_test, cov = false)))

@traitfn predict_y(
    model::TGP,
    X_test::AbstractMatrix,
) where {TGP<:AbstractGP; IsMultiOutput{TGP}} = predict_y(model, [X_test])

@traitfn predict_y(
    model::TGP,
    X_test::AbstractVector{<:AbstractMatrix},
) where {TGP<:AbstractGP; IsMultiOutput{TGP}} =
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
proba_y(model::AbstractGP{T}, X_test::AbstractVector{T}) where {T<:Real} =
    proba_y(model, reshape(X_test, :, 1))

"""
`proba_y(model::AbstractGP,X_test::AbstractMatrix)`

Return the probability distribution p(y_test|model,X_test) :

    - Tuple of vectors of mean and variance for regression
    - Vector of probabilities of y_test = 1 for binary classification
    - Dataframe with columns and probability per class for multi-class classification
"""
@traitfn function proba_y(
    model::TGP,
    X_test::AbstractMatrix,
) where {TGP<:AbstractGP; !IsMultiOutput{TGP}}
    μ_f, Σ_f = _predict_f(model, X_test, cov = true)
    pred = compute_proba(model.likelihood, μ_f, Σ_f)
end

@traitfn proba_y(
    model::TGP,
    X_test::AbstractMatrix,
) where {TGP<:AbstractGP; IsMultiOutput{TGP}} = proba_y(model, [X_test])

@traitfn function proba_y(
    model::TGP,
    X_test::AbstractVector{<:AbstractMatrix},
) where {TGP<:AbstractGP; IsMultiOutput{TGP}}
    μ_f, Σ_f = _predict_f(model, X_test, cov = true)
    preds = compute_proba.(model.likelihood, μ_f, Σ_f)
end

function proba_y(
    model::AbstractGP{T,<:MultiClassLikelihood},
    X_test::AbstractMatrix,
) where {T}
    μ_f, Σ_f = _predict_f(model, X_test, cov = true)
    μ_p = compute_proba(model.likelihood, μ_f, Σ_f)
end

compute_proba(
    l::Likelihood,
    μ::AbstractVector{<:AbstractVector},
    σ²::AbstractVector{<:AbstractVector},
) = compute_proba(l, first(μ), first(σ²))

function proba_y(
    model::MCGP{T,<:Likelihood,<:GibbsSampling},
    X_test::AbstractMatrix{T};
    nSamples::Int = 200,
) where {T<:Real}
    N_test = size(X_test, 1)
    f = _sample_f(model, X_test)
    nf = length(model.inference.sample_store[1])
    proba = [zeros(size(X_test, 1)) for i = 1:model.nLatent]
    sig_proba = [zeros(size(X_test, 1)) for i = 1:model.nLatent]
    for i = 1:nf
        for k = 1:model.nLatent
            proba[k], sig_proba[k] =
                (proba[k], sig_proba[k]) .+
                compute_proba(model.likelihood, getindex.(f, [i])[k], K̃[k])
        end
    end
    if model.nLatent == 1
        return (proba[1] / nf, sig_proba[1] / nf)
    else
        return (proba ./ nf, sig_proba ./ nf)
    end
end

#
# function proba_y(model::VGP{T,<:MultiClassLikelihood{T},<:GibbsSampling{T}},X_test::AbstractMatrix{T};nSamples::Int=200) where {T}
#     k_star = kernelmatrix.([X_test],[model.inference.x],model.kernel)
#     f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
#     k_starstar = kerneldiagmatrix.([X_test],model.kernel)
#     K̃ = k_starstar .- opt_diag.(k_star.*model.invKnn,k_star) .+ [zeros(size(X_test,1)) for i in 1:model.nLatent]
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
    l::Likelihood{T},
    μ::AbstractVector{T},
    σ²::AbstractVector{},
) where {T<:Real}
    @error "Non implemented for the likelihood $l"
end
