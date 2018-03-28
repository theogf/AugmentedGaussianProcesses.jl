"""
File treating all the prediction functions
"""


#### Computation of predictions with and without variance using the probit and logit link ####

function probitpredict(model::LinearModel,X_test)
    return model.Intercept ? [ones(Float64,size(X_test,1)) X_test]*model.μ : X_test*model.μ
end

function probitpredict(model::FullBatchModel,X_test)
    n = size(X_test,1)
    if model.TopMatrixForPrediction == 0
      model.TopMatrixForPrediction = model.invK*model.μ
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    return k_star*model.TopMatrixForPrediction
end

function probitpredict(model::SparseModel,X_test)
    n = size(X_test,1)
    if model.TopMatrixForPrediction == 0
      model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    return k_star*model.TopMatrixForPrediction
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
    n = size(X_test,1)
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = (model.invK*(eye(model.nSamples)-model.ζ*model.invK))
    end
    predic = zeros(n)
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    k_starstar = diagkernelmatrix(X_test,model.kernel)
    for i in 1:n
      predic[i] = cdf(Normal(),(dot(k_star[i,:],model.TopMatrixForPrediction))/(k_starstar[i] - dot(k_star[i,:],model.DownMatrixForPrediction*k_star[i,:]) + 1))
    end
    return predic
end

function probitpredictproba(model::SparseModel,X_test)
    n = size(X_test,1)
    if model.DownMatrixForPrediction == 0
      if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
      end
      model.DownMatrixForPrediction = (model.invKmm*(eye(model.m)-model.ζ*model.invKmm))
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    k_starstar = diagkernelmatrix(X_test,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    cov_fstar = k_starstar - sum(k_star.*transpose(model.DownMatrixForPrediction*k_star'),2)
    predic = broadcast((x,y)->cdf(Normal(),x/(y+1)),mean_fstar,cov_fstar)
    # for i in 1:n
    #   predic[i] = cdf(Normal(),m[i]/(k_starstar[i] - dot(k_star[i,:],model.DownMatrixForPrediction*k_star[i,:]) + 1))
    # end
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

function logitpredictproba(model::FullBatchModel,X_test)
    nPoints = size(X_test,1)
    if model.DownMatrixForPrediction == 0
      if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.μ
      end
      model.DownMatrixForPrediction = (model.invK*(eye(model.nSamples)-model.ζ*model.invK))
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    k_starstar = diagkernelmatrix(X_test,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction;
    cov_fstar = k_starstar-sum(k_star.*transpose(model.DownMatrixForPrediction*k_star'),2)
    @assert count(cov_fstar.<=0)==0  error("Covariance under 0")
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(mean_fstar[i],cov_fstar[i])
        f=function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,mean_fstar[i]-10*cov_fstar[i],mean_fstar[i]+10*cov_fstar[i])[1]
    end
    return predic
end

function logitpredictproba(model::SparseModel,X_test)
    nPoints = size(X_test,1)
    if model.DownMatrixForPrediction == 0
      if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
      end
      model.DownMatrixForPrediction = (model.invKmm*(eye(model.m)-model.ζ*model.invKmm))
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    k_starstar = diagkernelmatrix(X_test,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction;
    cov_fstar = k_starstar-sum(k_star.*transpose(model.DownMatrixForPrediction*k_star'),2)
    @assert count(cov_fstar.<=0)==0  error("Covariance under 0")
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(mean_fstar[i],cov_fstar[i])
        f=function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,mean_fstar[i]-10*cov_fstar[i],mean_fstar[i]+10*cov_fstar[i])[1]
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
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    return k_star*model.TopMatrixForPrediction
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
