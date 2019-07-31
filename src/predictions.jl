#File treating all the prediction functions

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
function predict_f(model::VGP,X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where T
    k_star = kernelmatrix.([X_test],[model.X],model.kernel)
    μf = k_star.*model.invKnn.*model.μ
    if !covf
        return model.nLatent == 1 ? μf[1] : μf
    end
    A = model.invKnn.*([I].-model.Σ.*model.invKnn)
    σ²f = []
    if fullcov
        k_starstar = kernelmatrix.([X_test],model.kernel)
        σ²f = Symmetric.(k_starstar .- k_star.*A.*transpose.(k_star) .+ convert(T,Jittering()).*[I])
    else
        k_starstar = kerneldiagmatrix.([X_test],model.kernel)
        σ²f = k_starstar .- opt_diag.(k_star.*A,k_star)
    end
    return model.nLatent == 1 ? (μf[1],σ²f[1]) : (μf,σ²f)
end

"""
Compute the mean of the predicted latent distribution of f on `X_test` for a sparse GP `model`
Return also the variance if `covf=true` and the full covariance if `fullcov=true`
"""
function predict_f(model::SVGP,X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where T
    k_star = kernelmatrix.([X_test],model.Z,model.kernel)
    μf = k_star.*model.invKmm.*model.μ
    if !covf
        return model.nLatent == 1 ? μf[1] : μf
    end
    A = model.invKmm.*([I].-model.Σ.*model.invKmm)
    if fullcov
        k_starstar = kernelmatrix.([X_test],model.kernel)
        σ²f = Symmetric.(k_starstar .- k_star.*A.*transpose.(k_star))
    else
        k_starstar = kerneldiagmatrix.([X_test],model.kernel)
        σ²f = k_starstar .- opt_diag.(k_star.*A,k_star)
    end
    return model.nLatent == 1 ? (μf[1],σ²f[1]) : (μf,σ²f)
end

function predict_f(model::VGP{T,<:Likelihood,<:GibbsSampling},X_test::AbstractMatrix{T};covf::Bool=true,fullcov::Bool=false) where {T}
    k_star = kernelmatrix.([X_test],[model.X],model.kernel)
    f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
    μf =  [vec(mean(hcat(f[k]...),dims=2)) for k in 1:model.nLatent]
    if !covf
        return model.nLatent == 1 ? μf[1] : μf
    end
    σ²f = []
    if fullcov
        k_starstar = kernelmatrix.([X_test],model.kernel)
        σ²f = Symmetric.(k_starstar .- k_star.*model.invKnn.*transpose.(k_star) .+  cov.(f))
    else
        k_starstar = kerneldiagmatrix.([X_test],model.kernel)
        σ²f = k_starstar .- opt_diag.(k_star.*model.invKnn,k_star) .+  diag.(cov.(f))
    end
    return model.nLatent == 1 ? (μf[1],σ²f[1]) : (μf,σ²f)
end

function predict_f(model::AbstractGP,X_test::AbstractVector{T};covf::Bool=false,fullcov::Bool=false) where T
    predict_f(model,reshape(X_test,length(X_test),1),covf=covf,fullcov=fullcov)
end

## Wrapper to predict vectors ##
function predict_y(model::AbstractGP,X_test::AbstractVector)
    return predict_y(model,reshape(X_test,length(X_test),1))
end

"""
`predict_y(model::AbstractGP{T,<:RegressionLikelihood},X_test::AbstractMatrix)`

Return the predictive mean of `X_test`
"""
function predict_y(model::AbstractGP{T,<:RegressionLikelihood},X_test::AbstractMatrix) where {T}
    return predict_f(model,X_test,covf=false)
end

"""
`predict_y(model::AbstractGP{T,<:ClassificationLikelihood},X_test::AbstractMatrix)`

Return the predicted most probable sign of `X_test`
"""
function predict_y(model::AbstractGP{T,<:ClassificationLikelihood},X_test::AbstractMatrix) where {T}
    return [sign.(f) for f in predict_f(model,X_test,covf=false)]
end

"""
`predict_y(model::AbstractGP{T,<:MultiClassLikelihood},X_test::AbstractMatrix)`

Return the predicted most probable class of `X_test`
"""
function predict_y(model::AbstractGP{T,<:MultiClassLikelihood},X_test::AbstractMatrix) where {T}
    n = size(X_test,1)
    μ_f = predict_f(model,X_test,covf=false)
    return [model.likelihood.class_mapping[argmax([μ[i] for μ in μ_f])] for i in 1:n]
end

"""
`predict_y(model::AbstractGP{T,<:EventLikelihood},X_test::AbstractMatrix)`

Return the expected number of events for the locations `X_test`
"""
function predict_y(model::AbstractGP{T,<:EventLikelihood},X_test::AbstractMatrix) where {T}
    n = size(X_test,1)
    μ_f = predict_f(model,X_test,covf=false)
    return model.likelihood.λ.*((x->logistic.(x)).(μ_f))
end

## Wrapper to return proba on vectors
function proba_y(model::AbstractGP,X_test::AbstractVector{T}) where {T<:Real}
    return proba_y(model,reshape(X_test,:,1))
end

"""
`proba_y(model::AbstractGP,X_test::AbstractMatrix)`

Return the probability distribution p(y_test|model,X_test) :

    - Tuple of vectors of mean and variance for regression
    - Vector of probabilities of y_test = 1 for binary classification
    - Dataframe with columns and probability per class for multi-class classification
"""
function proba_y(model::AbstractGP,X_test::AbstractMatrix)
    μ_f,Σ_f = predict_f(model,X_test,covf=true)
    compute_proba(model.likelihood,μ_f,Σ_f)
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

function proba_y(model::VGP{T,<:ClassificationLikelihood,<:GibbsSampling},X_test::AbstractMatrix{T};nSamples::Int=200) where {T<:Real}
    k_star = kernelmatrix.([X_test],[model.X],model.kernel)
    f = [[k_star[min(k,model.nPrior)]*model.invKnn[min(k,model.nPrior)]].*model.inference.sample_store[k] for k in 1:model.nLatent]
    k_starstar = kerneldiagmatrix.([X_test],model.kernel)
    K̃ = k_starstar .- opt_diag.(k_star.*model.invKnn,k_star) .+ [zeros(size(X_test,1)) for i in 1:model.nLatent]
    nf = length(model.inference.sample_store[1])
    proba = [zeros(size(X_test,1)) for i in 1:model.nLatent]
    for i in 1:nf
        proba .+= compute_proba(model.likelihood,getindex.(f,[i]),K̃)
    end
    if model.nLatent == 1
        return proba[1]/nf
    else
        return proba./nf
    end
end

function compute_proba(l::Likelihood{T},μ::AbstractVector{<:AbstractVector{T}},σ²::AbstractVector{<:AbstractVector{T}}) where {T<:Real}
    compute_proba.([l],μ,σ²)
end

### TODO Think about a better solution (general multi-likelihood problem)


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
