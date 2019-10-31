#File treating all the prediction functions

const pred_nodes,pred_weights = gausshermite(100) |> x->(x[1].*sqrt2,x[2]./sqrtπ)

function _predict_f(μ::Vector{T},Σ::Symmetric{T,Matrix{T}},invK::Symmetric{T,Matrix{T}},kernel::Kernel,X_test::AbstractMatrix{T₁},X::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T,T₁<:Real}
    k_star = kernelmatrix(X_test,X,kernel)
    μf = k_star*invK*μ
    if !covf
        return μf
    end
    A = invK*(I-Σ*invK)
    σ²f = []
    if fullcov
        k_starstar = kernelmatrix(X_test,kernel)
        σ²f = Symmetric(k_starstar - k_star*A*transpose(k_star))
    else
        k_starstar = kerneldiagmatrix(X_test,kernel)
        σ²f = k_starstar - opt_diag(k_star*A,k_star)
    end
    return μf,σ²f
end
"""
Compute the mean of the predicted latent distribution of `f` on `X_test` for the variational GP `model`

Return also the variance if `covf=true` and the full covariance if `fullcov=true`
"""
function predict_f(model::AbstractGP1,X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = get_σ_k(model)*kernelmatrix(model.f[1].kernel,X_test,get_X(model),obsdim=1)
    μf = k_star*(get_K(model)\get_μ(model))
    if !covf
        return μf
    end
    A = get_K(model)\(I-get_Σ(model)/get_K(model))
    if fullcov
        k_starstar = get_σ_k(model)*(kernelmatrix(model.f[1].kernel,X_test,obsdim=1)+T(jitter)*I)
        σ²f = Symmetric(k_starstar - k_star*A*transpose(k_star))
        return μf,σ²f
    else
        k_starstar = get_σ_k(model)*(kerneldiagmatrix(model.f[1].kernel,X_test,obsdim=1).+T(jitter))
        σ²f = k_starstar - opt_diag(k_star*A,k_star)
        return μf,σ²f
    end
end

function predict_f(model::AbstractGP,X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test],get_X(model),obsdim=1)
    μf = k_star.*(get_K(model).\get_μ(model))
    if !covf
        return μf
    end
    A = get_K(model).\([I].-get_Σ(model)./get_K(model))
    if fullcov
        k_starstar = get_σ_k(model).*(kernelmatrix.(get_kernel(model),[X_test],obsdim=1).+T(jitter)*[I])
        σ²f = Symmetric.(k_starstar .- k_star.*A.*transpose.(k_star))
        return μf,σ²f
    else
        k_starstar = get_σ_k(model).*(kerneldiagmatrix.(get_kernel(model),[X_test],obsdim=1).+T(jitter))
        σ²f = k_starstar .- opt_diag.(k_star.*A,k_star)
        return μf,σ²f
    end
end

function predict_f(model::MOSVGP,X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = get_σ_k(model).*kernelmatrix.(get_kernel(model),[X_test],get_X(model),obsdim=1)
    μf = k_star.*(get_K(model).\get_μ(model))
    μf = [[sum(vec(model.A[i,j,:]).*μf) for j in 1:model.nf_per_task[i]] for i in 1:model.nTask]
    if !covf
        return μf
    end
    A = get_K(model).\([I].-get_Σ(model)./get_K(model))
    if fullcov
        k_starstar = get_σ_k(model).*(kernelmatrix.(get_kernel(model),[X_test],obsdim=1).+T(jitter)*[I])
        Σf = - k_star.*A.*transpose.(k_star)
        Σf = [[sum(k_starstar.+vec(model.A[i,j,:]).^2 .*Σf) for j in 1:model.nf_per_task[i]] for i in 1:model.nTask]
        return μf,Σf
    else
        k_starstar = get_σ_k(model).*(kerneldiagmatrix.(get_kernel(model),[X_test],obsdim=1).+[T(jitter)*ones(T,size(X_test,1))])
        σ²f = - opt_diag.(k_star.*A,k_star)
        σ²f = [[sum(k_starstar.+vec(model.A[i,j,:]).^2 .*σ²f) for j in 1:model.nf_per_task[i]] for i in 1:model.nTask]
        return μf,σ²f
    end
end
# function predict_f(model::MCGP{T,<:Likelihood,<:GibbsSampling},X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
#     k_star = kernelmatrix.([X_test],[model.X],model.kernel)
#     f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
#     μf =  [vec(mean(hcat(f[k]...),dims=2)) for k in 1:model.nLatent]
#     if !covf
#         return model.nLatent == 1 ? μf[1] : μf
#     end
#     σ²f = []
#     if fullcov
#         k_starstar = kernelmatrix.([X_test],model.kernel)
#         σ²f = Symmetric.(k_starstar .- k_star.*model.invKnn.*transpose.(k_star) .+  cov.(f))
#     else
#         k_starstar = kerneldiagmatrix.([X_test],model.kernel)
#         σ²f = k_starstar .- opt_diag.(k_star.*model.invKnn,k_star) .+  diag.(cov.(f))
#     end
#     return model.nLatent == 1 ? (μf[1],σ²f[1]) : (μf,σ²f)
# end

function predict_f(model::AbstractGP,X_test::AbstractVector{T};covf::Bool=false,fullcov::Bool=false) where T
    predict_f(model,reshape(X_test,length(X_test),1),covf=covf,fullcov=fullcov)
end

## Wrapper to predict vectors ##
function predict_y(model::AbstractGP,X_test::AbstractVector)
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
function predict_y(model::AbstractGP{T,<:RegressionLikelihood},X_test::AbstractMatrix) where {T}
    return predict_y(model.likelihood,predict_f(model,X_test,covf=false))
end

predict_y(model::MOSVGP,X_test::AbstractMatrix) = predict_y.(model.likelihood,predict_f(model,X_test,covf=false))

predict_y(l::RegressionLikelihood,μ::AbstractVector{<:Real}) = μ
predict_y(l::RegressionLikelihood,μ::AbstractVector{<:AbstractVector})= first(μ)
predict_y(l::ClassificationLikelihood,μ::AbstractVector{<:Real}) = sign.(μ)
predict_y(l::ClassificationLikelihood,μ::AbstractVector{<:AbstractVector}) = sign.(first(μ))
predict_y(l::MultiClassLikelihood,μs::AbstractVector{<:AbstractVector}) = [l.class_mapping[argmax([μ[i] for μ in μs])] for i in 1:length(μs[1])]
predict_y(l::EventLikelihood,μ::AbstractVector{<:Real}) = expec_count(l,μ)
predict_y(l::EventLikelihood,μ::AbstractVector{<:AbstractVector}) = expec_count(l,first(μ))

## Wrapper to return proba on vectors ##
proba_y(model::AbstractGP,X_test::AbstractVector) = proba_y(model,reshape(X_test,:,1))

"""
`proba_y(model::AbstractGP,X_test::AbstractMatrix)`

Return the probability distribution p(y_test|model,X_test) :

    - Tuple of vectors of mean and variance for regression
    - Vector of probabilities of y_test = 1 for binary classification
    - Dataframe with columns and probability per class for multi-class classification
"""
function proba_y(model::AbstractGP,X_test::AbstractMatrix)
    μ_f,Σ_f = predict_f(model,X_test,covf=true)
    μ_p, σ²_p = compute_proba(model.likelihood,μ_f,Σ_f)
end

function proba_y(model::MOSVGP,X_test::AbstractMatrix)
    μ_f,Σ_f = predict_f(model,X_test,covf=true)
    μ_p, σ²_p = compute_proba.(model.likelihood,μ_f,Σ_f)
end

compute_proba(l::Likelihood,μ::AbstractVector{<:AbstractVector},σ²::AbstractVector{<:AbstractVector}) = compute_proba(l,first(μ),first(σ²))

function proba_y(model::VGP{T,<:Union{<:RegressionLikelihood{T},<:ClassificationLikelihood{T}},<:GibbsSampling},X_test::AbstractMatrix{T};nSamples::Int=200) where {T<:Real}
    N_test = size(X_test,1)
    k_star = kernelmatrix.([X_test],[model.X],model.kernel)
    f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
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
    k_star = kernelmatrix.([X_test],[model.X],model.kernel)
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
# function proba_y(model::VGP{T,<:ClassificationLikelihood,<:GibbsSampling},X_test::AbstractMatrix{T};nSamples::Int=200) where {T<:Real}
#     k_star = kernelmatrix.([X_test],[model.X],model.kernel)
#     f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
#     k_starstar = kerneldiagmatrix.([X_test],model.kernel)
#     K̃ = k_starstar .- opt_diag.(k_star.*model.invKnn,k_star) .+ [zeros(size(X_test,1)) for i in 1:model.nLatent]
#     nf = length(model.inference.sample_store[1])
#     proba = [zeros(size(X_test,1)) for i in 1:model.nLatent]
#     sig_proba = [zeros(size(X_test,1)) for i in 1:model.nLatent]
#     for i in 1:nf
#         prob, sig_prob = compute_proba(model.likelihood,getindex.(f,[i]),K̃)
#         proba .+= prob
#         sig_proba .+= sig_prob
#     end
#     if model.nLatent == 1
#         return (proba[1]/nf, sig_proba[1]/nf)
#     else
#         return (proba./nf, sig_proba./nf)
#     end
# end

# function compute_proba(l::Likelihood{T},μ::AbstractVector{<:AbstractVector{T}},σ²::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
#     compute_proba.(l,μ,σ²)
# end

function compute_proba(l::Likelihood{T},μ::AbstractVector{T},σ²::AbstractVector{}) where {T<:Real}
    @error "Non implemented for the likelihood $l"
end

# "Return the mean of likelihood p(y*=1|X,x*) via the probit link with a linear model"
# function probitpredictproba(model::LinearModel,X_test::AbstractArray{T}) where {T<:Real}
#     if model.Intercept
#       X_test = [ones(T,size(X_test,1)) X_test]
#     end
#     n = size(X_test,1)
#     pred = zeros(n)
#     for i in 1:nTest
#       pred[i] = cdf(Normal(),(dot(X_test[i,:],model.μ))/(dot(X_test[i,:],model.Σ*X_test[i,:])+1))
#     end
#     return pred
# end

"""Return the mean of likelihood p(y*=1|X,x*) via the probit link with a GP model"""
function probitpredictproba(model::AbstractGP,X_test::AbstractArray{<:Real})
    m_f,cov_f = predict_f(model,X_test,covf=true)
    return broadcast((m,c)->cdf(Normal(),m/(c+1)),m_f,cov_f)
end

"Return the modified softmax likelihood given the latent functions"
function sigma_max(f::Vector{T},index::Integer) where {T<:Real}
    return logit(f[index])/sum(logit.(f))
end

"Return the modified softmax likelihood given the array of 'σ' and their sum (can be given via sumsig)"
function mod_soft_max(σ::Vector{T},sumsig::T=zero(T)) where {T<:Real}
    return sumsig == 0 ? σ./(sum(σ)) : σ./sumsig
end


"Return the gradient of the modified softmax likelihood given 'σ' and their sum (can be given via sumsig)"
function grad_mod_soft_max(σ::Array{T,1},sumsig::T=zero(T)) where {T<:Real}
    sumsig = sumsig == 0 ? sum(σ) : sumsig
    shortened_sum = sumsig.-σ
    sum_square = sumsig^2
    base_grad = (σ-(σ.^2))./sum_square
    n = size(σ,1)
    grad = zeros(n,n)
    for i in 1:n
        for j in 1:n
            if i==j
                grad[i,i] = shortened_sum[i]*base_grad[i]
            else
                grad[i,j] = -σ[i]*base_grad[j]
            end
        end
    end
    return grad
end

"Return the hessian of the modified softmax likelihood given 'σ' and their sum (can be given via sumsig)"
function hessian_mod_soft_max(σ::AbstractVector{T},sumsig::T=zero(T)) where {T<:Real}
    sumsig = sumsig == 0 ? sum(σ) : sumsig
    shortened_sum = sumsig.-σ
    sum_square = sumsig^2
    sum_cube = sumsig^3
    base_grad = (σ-σ.^2).*((1.0.-2.0.*σ).*sum_square-2.0.*(σ-σ.^2)*sumsig)./sum_cube
    n = size(σ,1)
    grad = zeros(n,n)
    for i in 1:n
        for j in 1:n
            if i==j
                grad[i,i] = shortened_sum[i]*base_grad[i]
            else
                grad[i,j] = -σ[i]*base_grad[j]
            end
        end
    end
    return grad
end
