"""
File treating all the prediction functions
"""


#### Computation of predictions with and without variance using the probit and logit link ####
# function fstar(model::LinearModel,X_test,cov::Bool=true)
# end

function fstar(model::FullBatchModel,X_test,covf::Bool=true)
    if model.DownMatrixForPrediction == 0
        if covf && model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = (model.invK*(eye(model.nSamples)-model.ζ*model.invK))
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        k_starstar = diagkernelmatrix(X_test,model.kernel)
        cov_fstar = k_starstar - sum(k_star.*transpose(model.DownMatrixForPrediction*k_star'),2)
        return mean_fstar,cov_fstar
    end
end

function fstar(model::SparseModel,X_test,covf::Bool=true)
    if model.DownMatrixForPrediction == 0
        if covf && model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.Kmm\model.μ
        end
      model.DownMatrixForPrediction = (model.Kmm\(eye(model.nFeatures)-model.ζ/model.Kmm))
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        k_starstar = diagkernelmatrix(X_test,model.kernel)
        cov_fstar = k_starstar - sum(k_star.*transpose(model.DownMatrixForPrediction*k_star'),2)
        return mean_fstar,cov_fstar
    end
end

function probitpredict(model::LinearModel,X_test)
    return model.Intercept ? [ones(Float64,size(X_test,1)) X_test]*model.μ : X_test*model.μ
end

function probitpredict(model::FullBatchModel,X_test)
    return fstar(model,X_test,covf=false)
end

function probitpredict(model::SparseModel,X_test)
    return fstar(model,X_test,covf=false)
end

function probitpredictproba(model::LinearModel,X_test)
    if model.Intercept
      X_test = [ones(Float64,size(X_test,1)) X_test]
    end
    n = size(X_test,1)
    predic = zeros(n)
    for i in 1:n
      predic[i] = cdf(Normal(),(dot(X_test[i,:],model.μ))/(dot(X_test[i,:],model.ζ*X_test[i,:])+1))
    end
    return predic
end

function probitpredictproba(model::FullBatchModel,X_test)
    m_f,cov_f = fstar(model,X_test,covf=true)
    return broadcast((m,c)->cdf(Normal(),m/(c+1)),m_f,cov_f)
end

function probitpredictproba(model::SparseModel,X_test)
    m_f,cov_f = fstar(model,X_test,covf=true)
    predic = broadcast((m,c)->cdf(Normal(),m/(c+1)),m_f,cov_f)
    return predic
end

function logit(x)
    return 1./(1+exp.(-x))
end

function logitpredict(model::GPModel,X_test)
    y_predic = logitpredictproba(model,X_test)
    y_predic[y_predic.>0.5] = 1; y_predic[y_predic.<=0.5] = -1
    return y_predic
end

function logitpredictproba(model::GPModel,X_test)
    m_f,cov_f = fstar(model,X_test,covf=true)
    @assert minimum(cov_f)<0  error("Covariance under 0")
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(mean_fstar[i],cov_fstar[i])
        f=function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,m_f[i]-10*cov_f[i],m_f[i]+10*cov_f[i])[1]
    end
    return predic
end

function regpredict(model::GPRegression,X_test)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.y
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    return k_star*model.TopMatrixForPrediction
end

function regpredict(model::SparseGPRegression,X_test)
    return fstar(model,X_test,covf=false)
end

#Return the mean and variance of the predictive distribution of f*
function regpredictproba(model::FullBatchModel,X_test)
    n = size(X_test,1)
    ksize = model.nSamples
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = -(model.invK*(eye(ksize)-model.ζ*model.invK))
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    k_starstar = diagkernelmatrix(X_test,model.kernel)
    meanfstar = k_star*model.TopMatrixForPrediction
    covfstar = k_starstar - k_star*model.DownMatrixForPrediction*transpose(k_star)
    return meanfstar,covfstar
end

#Return the mean and variance of the predictive distribution of f*
function regpredictproba(model::SparseModel,X_test)
    n = size(X_test,1)
    ksize = model.nSamples
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invKmm*model.μ
        end
      model.DownMatrixForPrediction = -(model.invKmm*(eye(ksize)-model.ζ*model.invKmm))
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    k_starstar = diagkernelmatrix(X_test,model.kernel)
    meanfstar = k_star*model.TopMatrixForPrediction
    covfstar = k_starstar + diag(A*(model.ζ*model.invK-eye(model.nSamples))*transpose(k_star))
    return meanfstar,covfstar
end
