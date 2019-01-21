#File treating all the prediction functions

"""
Compute the mean of the predicted latent distribution of f on X_test for full GP models
Return also the variance if `covf=true`
"""
function fstar(model::FullBatchModel,X_test::AbstractArray;covf::Bool=true) where T
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.μ
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    end
    model.DownMatrixForPrediction = (model.invK*(I-model.Σ*model.invK))
    cov_fstar = kerneldiagmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
    return mean_fstar,cov_fstar
end

"""
Compute the mean of the predicted latent distribution of f on X_test for sparse GP models
Return also the variance if `covf=true`
"""
function fstar(model::SparseModel,X_test::AbstractArray;covf::Bool=true)
    model.TopMatrixForPrediction = model.invKmm*model.μ
    if covf
        model.DownMatrixForPrediction = model.invKmm*(I-model.Σ*model.invKmm)
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    end
    cov_fstar = kerneldiagmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
    return mean_fstar,cov_fstar
end

"""
Compute the mean of the predicted latent distribution of f on X_test for online GP models
Return also the variance if `covf=true`
"""
function fstar(model::OnlineGPModel,X_test::AbstractArray;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = (model.invKmm*(I-model.Σ*model.invKmm))
    end
    k_star = kernelmatrix(X_test,model.kmeansalg.centers,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        cov_fstar = kerneldiagmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
        return mean_fstar,cov_fstar
    end
end

"""
Compute the mean of the predicted latent distribution of f on X_test for GP regression
Return also the variance if `covf=true`
"""
function fstar(model::BatchGPRegression,X_test::AbstractArray;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.y
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = model.invK
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        cov_fstar = kerneldiagmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
        return mean_fstar,cov_fstar
    end
end

"""
Compute the mean of the predicted latent distribution of f on X_test for Multiclass GP models
Return also the variance if `covf=true`
"""
function fstar(model::MultiClassGPModel,X_test::AbstractArray;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = broadcast((mu,invK)->invK*mu,model.μ,model.invK)
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = broadcast((Σ,invK)->invK*(I-Σ*invK),model.Σ,model.invK)
    end
    if model.IndependentGPs
        k_star = [KernelModule.kernelmatrix(X_test,model.X,model.kernel[i]) for i in 1:model.K]
    else
        k_star = [KernelModule.kernelmatrix(X_test,model.X,model.kernel[1])]
    end
    mean_fstar = broadcast((k_s,m)->k_s*m,k_star,model.TopMatrixForPrediction)
    if !covf
        return mean_fstar
    else
        if model.IndependentGPs
            k_starstar = [kerneldiagmatrix(X_test,model.kernel[i]) for i in 1:model.K]
        else
            k_starstar = [kerneldiagmatrix(X_test,model.kernel[1])]
        end
        cov_fstar = broadcast((k_ss,k_s,x)->(k_ss .- sum((k_s*x).*k_s,dims=2)[:]),k_starstar,k_star,model.DownMatrixForPrediction)
        return mean_fstar,cov_fstar
    end
end

"""
Compute the mean of the predicted latent distribution of f on X_test for multiclass sparse GP models
Return also the variance if `covf=true`
"""
function fstar(model::Union{SparseMultiClass,SparseLogisticSoftMaxMultiClass},X_test::AbstractArray;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm.*model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
        model.DownMatrixForPrediction = broadcast((Σ,invKmm)->invKmm*(I-Σ*invKmm),model.Σ,model.invKmm)
    end
    k_star = broadcast((points,kernel)->KernelModule.kernelmatrix(X_test,points,kernel),model.inducingPoints,model.kernel)
    # k_star = broadcast((points,kernel)->MLKernels.kernelmatrix(kernel,X_test,points),model.inducingPoints,model.altkernel)
    mean_fstar = k_star.*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    end
    k_starstar = kerneldiagmatrix.([X_test],model.kernel)
    # cov_fstar = [zeros(T,size(X_test,1)) for _ in 1:model.K]
    cov_fstar = broadcast((x,ks,kss)->(kss .- sum((ks*x).*ks,dims=2)[:]),model.DownMatrixForPrediction,k_star,k_starstar)
    return mean_fstar,cov_fstar
end

"Return the predicted class {-1,1} with a linear model via the probit link"
function probitpredict(model::LinearModel,X_test::AbstractArray{T}) where {T<:Real}
    return sign.((model.Intercept ? [ones(T,size(X_test,1)) X_test]*model.μ : X_test*model.μ).-0.5)
end

"Return the predicted class {-1,1} with a GP model via the probit link"
function probitpredict(model::GPModel,X_test::AbstractArray)
    return sign.(fstar(model,X_test,covf=false).-0.5)
end

"Return the mean of likelihood p(y*=1|X,x*) via the probit link with a linear model"
function probitpredictproba(model::LinearModel,X_test::AbstractArray{T}) where {T<:Real}
    if model.Intercept
      X_test = [ones(T,size(X_test,1)) X_test]
    end
    n = size(X_test,1)
    pred = zeros(n)
    for i in 1:nTest
      pred[i] = cdf(Normal(),(dot(X_test[i,:],model.μ))/(dot(X_test[i,:],model.Σ*X_test[i,:])+1))
    end
    return pred
end

"""Return the mean of likelihood p(y*=1|X,x*) via the probit link with a GP model"""
function probitpredictproba(model::GPModel,X_test::AbstractArray)
    m_f,cov_f = fstar(model,X_test,covf=true)
    return broadcast((m,c)->cdf(Normal(),m/(c+1)),m_f,cov_f)
end

"""Return likelihood equivalent to SVM hinge loss"""
function svmlikelihood(x)
    pos = svmpseudolikelihood(x)
    return pos./(pos.+svmpseudolikelihood(-x))
end

"""Return the pseudo likelihood of the SVM hinge loss"""
function svmpseudolikelihood(x)
    return exp.(-2.0*max.(1.0.-x,0))
end


"""Return the point estimate of the likelihood of class y=1 via the SVM likelihood"""
function svmpredict(model::GPModel,X_test::AbstractArray)
    return sign.(fstar(model,X_test,covf=false))
end

"""Return the likelihood of class y=1 via the SVM likelihood"""
function svmpredictproba(model::GPModel,X_test::AbstractArray)
    m_f,cov_f = fstar(model,X_test,covf=true)
    nTest = length(m_f)
    pred = zero(m_f)
    for i in 1:nTest
        if cov_f[i] <= 0
            pred[i] = svmlikelihood(m_f[i])
        else
            d = Normal(m_f[i],sqrt(cov_f[i]))
            pred[i] = expectation(svmlikelihood,d)
        end
    end
    return pred
end

"""Return logit(x)"""
function logit(x)
    return 1.0./(1.0.+exp.(-x))
end


"""Return the predicted class {-1,1} with a GP model via the logit link"""
function logitpredict(model::GPModel,X_test::AbstractArray)
    return sign.(fstar(model,X_test,covf=false))
end

"""Return the mean of likelihood p(y*=1|X,x*) via the logit link with a GP model"""
function logitpredictproba(model::GPModel,X_test::AbstractArray)
    m_f,cov_f = fstar(model,X_test,covf=true)
    nTest = length(m_f)
    pred = zero(m_f)
    for i in 1:nTest
        if cov_f[i] <= 0
            pred[i] = logit(m_f[i])
        else
            d = Normal(m_f[i],sqrt(cov_f[i]))
            pred[i] = expectation(logit,d)
        end
    end
    return pred
end

"""Return the mean of the predictive distribution of f"""
function regpredict(model::BatchGPRegression,X_test::AbstractArray)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.y
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    return k_star*model.TopMatrixForPrediction
end

"""Return the mean of the predictive distribution of f"""
function regpredict(model::GPModel,X_test::AbstractArray)
    return fstar(model,X_test,covf=false)
end

"""Return the mean and variance of the predictive distribution of f"""
function regpredictproba(model::GPModel,X_test::AbstractArray)
    m_f,cov_f =  fstar(model,X_test,covf=true)
    cov_f .+= model.gnoise
    return m_f,cov_f
end

"""Return the mean of the predictive distribution of f"""
function studenttpredict(model::GPModel,X_test::AbstractArray)
    return fstar(model,X_test,covf=false)
end


"""Return the mean and variance of the predictive distribution of f"""
function studenttpredictproba(model::GPModel,X_test::AbstractArray)
    return fstar(model,X_test,covf=true)
end

"Compute the mean and variance using MC integration"
function studentpredictprobamc(model::GPModel,X_test::AbstractArray{T};nSamples=100) where {T<:Real}
    m_f,cov_f = fstar(model,X_test,covf=true)
    nTest = length(m_f)
    mean_pred = zero(m_f)
    var_pred = zero(m_f)
    st = TDist(model.ν)
    temp_array = zeros(T,nSamples)
    for i in 1:nTest
        if cov_f[i] <= 1e-3
            pyf =  LocationScale(m_f[i],1.0,st)
            for j in 1:nSamples
                temp_array[j] = rand(pyf)
            end
        else
            d = Normal(m_f[i],sqrt(cov_f[i]))
            for j in 1:nSamples
                temp_array[j] = rand(LocationScale(rand(d),1.0,st))
            end
        end
        mean_pred[i] = mean(temp_array); var_pred[i] = cov(temp_array)
    end
    return mean_pred,var_pred
end

function multiclasspredict(model::MultiClassGPModel,X_test::AbstractArray{T},likelihood::Bool=false) where {T<:Real}
    n=size(X_test,1);
    m_f = fstar(model,X_test,covf=false)
    if !likelihood
        return [model.class_mapping[argmax([mu[i] for mu in m_f])] for i in 1:n]
    end
    return compute_proba(model,m_f)
end

function compute_proba(model::Union{MultiClass,SparseMultiClass},m_f::Vector{Vector{T}}) where T
    n = length(m_f)
    σ = hcat(logit.(m_f)...); σ = [σ[i,:] for i in 1:n]
    normsig = sum.(σ); y = mod_soft_max.(σ,normsig)
    pred = zeros(Int64,n)
    value = zeros(T,n)
    for i in 1:n
        res = findmax(y[i]);
        pred[i]=res[2];
        value[i]=res[1]
    end
    return model.class_mapping[pred],value
end

function compute_proba(model::Union{SoftMaxMultiClass,SparseLogisticSoftMaxMultiClass},m_f::Vector{Vector{T}}) where T
    n = length(m_f[1])
    m_f = hcat(m_f...); y = [softmax(m_f[i,:]) for i in 1:n]
    pred = zeros(Int64,n)
    value = zeros(T,n)
    for i in 1:n
        res = findmax(y[i]);
        pred[i]=res[2];
        value[i]=res[1]
    end
    return model.class_mapping[pred],value
end

function compute_proba(model::LogisticSoftMaxMultiClass,m_f::Vector{Vector{T}}) where T
    n = length(m_f[1])
    m_f = hcat(m_f...); y = [logisticsoftmax(m_f[i,:]) for i in 1:n]
    pred = zeros(Int64,n)
    value = zeros(T,n)
    for i in 1:n
        res = findmax(y[i]);
        pred[i]=res[2];
        value[i]=res[1]
    end
    return model.class_mapping[pred],value
end

function multiclasspredictlaplace(model::Union{MultiClass,SparseMultiClass},X_test::Array{T,N},covf::Bool=false) where {T,N}
    n = size(X_test,1)
    m_f,cov_f = fstar(model,X_test)
    σ = hcat(logit.(m_f)...)
    σ = [σ[i,:] for i in 1:n]
    cov_f = hcat(cov_f...)
    cov_f = [cov_f[i,:] for i in 1:n]
    normsig = sum.(σ)
    h = mod_soft_max.(σ,normsig)
    hess_h = hessian_mod_soft_max.(σ,normsig)
    m_predic = broadcast(m->max.(m,eps(T)),h.+0.5*broadcast((hess,cov)->(hess*cov),hess_h,cov_f))
    m_predic ./= sum.(m_predic)
    if !covf
        return DataFrame(hcat(m_predic...)',Symbol.(model.class_mapping))
        # return [m[model.class_mapping] for m in m_predic]
    end
    grad_h = grad_mod_soft_max.(σ,normsig)
    cov_predic = broadcast((grad,hess,cov)->(grad.^2*cov-0.25*hess.^2*(cov.^2)),grad_h,hess_h,cov_f)
    return DataFrame(hcat(hcat(m_predic...)',hcat(cov_predic...)'),[Symbol.(string.(model.class_mapping).+"_μ"),Symbol.(string.(model.class_mapping).+"_σ")])
    # return [m[model.class_mapping] for m in m_predic] ,[cov[model.class_mapping] for cov in cov_predic]
end

function multiclasspredictproba(model::SoftMaxMultiClass,X_test::Array{T,N},covf::Bool=false) where {T,N}
    n = size(X_test,1)
    m_f,cov_f = fstar(model,X_test)
    m_f = hcat(m_f...)
    m_f = [m_f[i,:] for i in 1:n]
    cov_f = hcat(cov_f...)
    cov_f = [cov_f[i,:] for i in 1:n]
    m_predic = zeros(n,model.K)
    nSamples = 200
    for i in 1:n
        p = MvNormal(m_f[i],sqrt.(max.(eps(T),cov_f[i])))
        for _ in 1:nSamples
            m_predic[i,:] += softmax(rand(p))/nSamples
        end
    end
    return DataFrame(m_predic,Symbol.(model.class_mapping))
end


function multiclasspredictproba(model::Union{LogisticSoftMaxMultiClass,SparseLogisticSoftMaxMultiClass},X_test::Array{T,N},covf::Bool=false) where {T,N}
    n = size(X_test,1)
    m_f,cov_f = fstar(model,X_test)
    m_f = hcat(m_f...)
    m_f = [m_f[i,:] for i in 1:n]
    cov_f = hcat(cov_f...)
    cov_f = [cov_f[i,:] for i in 1:n]
    m_predic = zeros(n,model.K)
    nSamples = 200
    for i in 1:n
        p = MvNormal(m_f[i],sqrt.(max.(eps(T),cov_f[i])))
        for _ in 1:nSamples
            m_predic[i,:] += logisticsoftmax(rand(p))/nSamples
        end
    end
    return DataFrame(m_predic,Symbol.(model.class_mapping))
end


function multiclasspredictproba(model::Union{MultiClass,SparseMultiClass},X_test::Array{T,N},covf::Bool=false;nSamples::Integer=200) where {T,N}
    n = size(X_test,1)
    m_f,cov_f = fstar(model,X_test)
    m_f = hcat(m_f...)
    m_f = [m_f[i,:] for i in 1:n]
    cov_f = hcat(cov_f...)
    cov_f = [cov_f[i,:] for i in 1:n]
    m_predic = zeros(n,model.K)
    for i in 1:n
        p = MvNormal(m_f[i],sqrt.(max.(eps(T),cov_f[i])))
        for _ in 1:nSamples
            m_predic[i,:] += logisticsoftmax(rand(p))/nSamples
        end
    end
    return DataFrame(m_predic,Symbol.(model.class_mapping))
end

"Return the modified softmax likelihood given the latent functions"
function sigma_max(f::Vector{T},index::Integer) where {T}
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
