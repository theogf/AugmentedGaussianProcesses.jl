#File treating all the prediction functions

const pred_nodes,pred_weights = gausshermite(100) |> x->(x[1].*sqrt2,x[2]./sqrtπ)

"""
Compute the mean of the predicted latent distribution of `f` on `X_test` for the variational GP `model`

Return also the diagonal variance if `covf=true` and the full covariance if `fullcov=true`
"""
function _predict_f(model::AbstractGP{T},X_test::AbstractMatrix{<:Real};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test],get_Z(model))
    μf = k_star.*(get_K(model).\get_μ(model))
    if !covf
        return (μf,)
    end
    A = get_K(model).\([I].-get_Σ(model)./getproperty.(get_K(model),:mat))
    if fullcov
        k_starstar = get_σ_k(model).*(kernelmatrix.(get_kernel(model),[X_test]).+T(jitter)*[I])
        Σf = Symmetric.(k_starstar .- k_star.*A.*transpose.(k_star))
        return μf,Σf
    else
        k_starstar = get_σ_k(model).*(kerneldiagmatrix.(get_kernel(model),[X_test]).+[T(jitter)*ones(T,size(X_test,1))])
        σ²f = k_starstar .- opt_diag.(k_star.*A,k_star)
        return μf,σ²f
    end
end

function _predict_f(model::GP{T},X_test::AbstractMatrix{<:Real};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test],get_Z(model))
    μf = k_star.*mean_f(model)
    if !covf
        return (μf,)
    end
    A = [inv(model.f[1].K+model.likelihood.σ²*I).mat]
    if fullcov
        k_starstar = get_σ_k(model).*(kernelmatrix.(get_kernel(model),[X_test]).+T(jitter)*[I])
        Σf = Symmetric.(k_starstar .- k_star.*A.*transpose.(k_star))
        return μf,Σf
    else
        k_starstar = get_σ_k(model).*(kerneldiagmatrix.(get_kernel(model),[X_test]).+[T(jitter)*ones(T,size(X_test,1))])
        σ²f = k_starstar .- opt_diag.(k_star.*A,k_star)
        return μf,σ²f
    end
end

function _predict_f(model::MOSVGP{T},X_test::AbstractMatrix{<:Real};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test],get_Z(model))
    μf = k_star.*(get_K(model).\get_μ(model))
    μf = [[sum(vec(model.A[i,j,:]).*μf) for j in 1:model.nf_per_task[i]] for i in 1:model.nTask]
    if !covf
        return (μf,)
    end
    A = get_K(model).\([I].-get_Σ(model)./get_K(model))
    if fullcov
        k_starstar = get_σ_k(model).*(kernelmatrix.(get_kernel(model),[X_test]).+T(jitter)*[I])
        Σf = k_starstar .-  k_star.*A.*transpose.(k_star)
        Σf = [[sum(vec(model.A[i,j,:]).^2 .*Σf) for j in 1:model.nf_per_task[i]] for i in 1:model.nTask]
        return μf,Σf
    else
        k_starstar = get_σ_k(model).*(kerneldiagmatrix.(get_kernel(model),[X_test]).+[T(jitter)*ones(T,size(X_test,1))])
        σ²f = k_starstar .- opt_diag.(k_star.*A,k_star)
        σ²f = [[sum(vec(model.A[i,j,:]).^2 .*σ²f) for j in 1:model.nf_per_task[i]] for i in 1:model.nTask]
        return μf,σ²f
    end
end

function _predict_f(model::MCGP{T,<:Likelihood,<:GibbsSampling},X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test],get_Z(model))
    f = _sample_f(model,X_test,k_star)
    μf = Tuple(vec(mean(f[k],dims=2)) for k in 1:model.nLatent)
    if !covf
        return (μf,)
    end
    if fullcov
        k_starstar = get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test])
        Σf = Symmetric.(k_starstar .- invquad.(get_K(model),k_star) .+  cov.(f))
        return μf, Σf
    else
        global k_starstar = get_σ_k(model).*kerneldiagmatrix.(get_kernel(model),[X_test]) .+ [T(jitter)*ones(T,size(X_test,1))]
        σ²f = k_starstar .- opt_diag.(k_star,get_K(model).\k_star) .+  diag.(cov.(f,dims=2))
        return μf,σ²f
    end
end

function _sample_f(model::MCGP{T,<:Likelihood,<:GibbsSampling},X_test::AbstractMatrix{T},k_star=get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test],get_Z(model))) where {T}
    return f = [k_star[k]*(model.f[k].K\model.inference.sample_store[:,:,k]') for k in 1:model.nLatent]
end

predict_f(model::AbstractGP{T},X_test::AbstractVector{T};covf::Bool=false,fullcov::Bool=false) where {T} = predict_f(model,reshape(X_test,length(X_test),1),covf=covf,fullcov=fullcov)

predict_f(model::AbstractGP{T,L,I,TGP,1},X_test::AbstractMatrix{T};covf::Bool=false,fullcov::Bool=false) where {T,L,I,TGP} = first.(_predict_f(model,X_test;covf=covf,fullcov=fullcov))

predict_f(model::AbstractGP,X_test::AbstractMatrix{T};covf::Bool=false,fullcov::Bool=false) where {T,L,I,TGP} = _predict_f(model,X_test;covf=covf,fullcov=fullcov)

## Wrapper to predict vectors ##
function predict_y(model::AbstractGP{T},X_test::AbstractVector{T}) where {T}
    return predict_y(model,reshape(X_test,:,1))
end

"""
`predict_y(model::AbstractGP,X_test::AbstractMatrix)`

Return
    - the predictive mean of `X_test` for regression
    - the sign of `X_test` for classification
    - the most likely class for multi-class classification
    - the expected number of events for an event likelihood
"""
function predict_y(model::AbstractGP{T},X_test::AbstractMatrix{T}) where {T}
    return predict_y(model.likelihood,_predict_f(model,X_test,covf=false)[1])
end

predict_y(model::MOSVGP,X_test::AbstractMatrix) = predict_y.(model.likelihood,_predict_f(model,X_test,covf=false))

predict_y(l::RegressionLikelihood,μ::AbstractVector{<:Real}) = μ
predict_y(l::RegressionLikelihood,μ::AbstractVector{<:AbstractVector}) = first(μ)
predict_y(l::ClassificationLikelihood,μ::AbstractVector{<:Real}) = sign.(μ)
predict_y(l::ClassificationLikelihood,μ::AbstractVector{<:AbstractVector}) = sign.(first(μ))
predict_y(l::MultiClassLikelihood,μs::AbstractVector{<:AbstractVector{<:Real}}) = [l.class_mapping[argmax([μ[i] for μ in μs])] for i in 1:length(μs[1])]
predict_y(l::MultiClassLikelihood,μs::AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}}) = predict_y(l,first(μs))
predict_y(l::EventLikelihood,μ::AbstractVector{<:Real}) = expec_count(l,μ)
predict_y(l::EventLikelihood,μ::AbstractVector{<:AbstractVector}) = expec_count(l,first(μ))

## Wrapper to return proba on vectors ##
proba_y(model::AbstractGP{T},X_test::AbstractVector{T}) where {T} = proba_y(model,reshape(X_test,:,1))

"""
`proba_y(model::AbstractGP,X_test::AbstractMatrix)`

Return the probability distribution p(y_test|model,X_test) :

    - Tuple of vectors of mean and variance for regression
    - Vector of probabilities of y_test = 1 for binary classification
    - Dataframe with columns and probability per class for multi-class classification
"""
function proba_y(model::AbstractGP,X_test::AbstractMatrix)
    μ_f,Σ_f = _predict_f(model,X_test,covf=true)
    pred = compute_proba(model.likelihood,μ_f,Σ_f)
end

function proba_y(model::MOSVGP,X_test::AbstractMatrix)
    μ_f,Σ_f = _predict_f(model,X_test,covf=true)
    preds = compute_proba.(model.likelihood,μ_f,Σ_f)
end

function proba_y(model::AbstractGP{T,<:MultiClassLikelihood},X_test::AbstractMatrix) where {T}
    μ_f,Σ_f = _predict_f(model,X_test,covf=true)
    μ_p = compute_proba(model.likelihood,μ_f,Σ_f)
end

compute_proba(l::Likelihood,μ::AbstractVector{<:AbstractVector},σ²::AbstractVector{<:AbstractVector}) = compute_proba(l,first(μ),first(σ²))

function proba_y(model::MCGP{T,<:Union{<:RegressionLikelihood{T},<:ClassificationLikelihood{T}},<:GibbsSampling},X_test::AbstractMatrix{T};nSamples::Int=200) where {T<:Real}
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

function proba_y(model::VGP{T,<:MultiClassLikelihood{T},<:GibbsSampling{T}},X_test::AbstractMatrix{T};nSamples::Int=200) where {T}
    k_star = kernelmatrix.([X_test],[model.inference.x],model.kernel)
    f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
    k_starstar = kerneldiagmatrix.([X_test],model.kernel)
    K̃ = k_starstar .- opt_diag.(k_star.*model.invKnn,k_star) .+ [zeros(size(X_test,1)) for i in 1:model.nLatent]
    nf = length(model.inference.sample_store[1])
    proba = zeros(size(X_test,1),model.nLatent)
    labels = Array{Symbol}(undef,model.nLatent)
    for i in 1:nf
        res = compute_proba(model.likelihood,getindex.(f,[i]),K̃,nSamples)
        if i ==  1
            labels = names(res)
        end
        proba .+= Matrix(res)
    end
    return DataFrame(proba/nf,labels)
end
#
function compute_proba(l::Likelihood{T},μ::AbstractVector{T},σ²::AbstractVector{}) where {T<:Real}
    @error "Non implemented for the likelihood $l"
end
