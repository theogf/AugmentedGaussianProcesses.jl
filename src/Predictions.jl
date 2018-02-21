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
    k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    return k_star*model.TopMatrixForPrediction
end

function probitpredict(model::SparseModel,X_test)
    n = size(X_test,1)
    if model.TopMatrixForPrediction == 0
      model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.inducingPoints)
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
    ksize = model.nSamples
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = -(model.invK*(eye(ksize)-model.ζ*model.invK))
    end
    predic = zeros(n)
    k_star = zeros(ksize)
    k_starstar = 0
    for i in 1:n
      for j in 1:ksize
        k_star[j] = model.Kernel_function(model.X[j,:],X_test[i,:])
      end
      k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])
      predic[i] = cdf(Normal(),(dot(k_star,model.TopMatrixForPrediction))/(k_starstar + dot(k_star,model.DownMatrixForPrediction*k_star) + 1))
    end
    return predic
end

function probitpredictproba(model::SparseModel,X_test)
    n = size(X_test,1)
    ksize = model.m
    if model.DownMatrixForPrediction == 0
      if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
      end
      model.DownMatrixForPrediction = -(model.invKmm*(eye(ksize)-model.ζ*model.invKmm))
    end
    predic = zeros(n)
    k_star = zeros(ksize)
    k_starstar = 0
    for i in 1:n
      for j in 1:ksize
        k_star[j] = model.Kernel_function(model.inducingPoints[j,:],X_test[i,:])
      end
      k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])
      predic[i] = cdf(Normal(),(dot(k_star,model.TopMatrixForPrediction))/(k_starstar + dot(k_star,model.DownMatrixForPrediction*k_star) + 1))
    end
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
    K_starN = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    K_starstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    m = K_starN*model.invK*model.μ;
    cov = K_starstar+sum((K_starN*model.invK).*transpose((model.ζ*model.invK-eye(model.nFeatures))*transpose(K_starN)),2)
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(m[i],cov[i])
        f=function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,m[i]-10*cov[i],m[i]+10*cov[i])[1]
    end
    return predic
end

function logitpredictproba(model::SparseModel,X_test)
    nPoints = size(X_test,1)
    ksize = model.m
    K_starM = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.inducingPoints)
    K_starstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    m = K_starM*model.invKmm*model.μ;
    cov = K_starstar+sum((K_starM*model.invKmm).*transpose((model.ζ*model.invKmm-eye(model.nFeatures))*transpose(K_starM)),2)
    if count(cov.<=0)>0
        error("Covariance under 0, params are $(broadcast(getfield,model.Kernels,:param)) and coeffs $(broadcast(getfield,model.Kernels,:coeff))")
    end
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(m[i],cov[i])
        f= function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,m[i]-10*cov[i],m[i]+10*cov[i])[1]
    end
    return predic
end

#Return the mean and variance of the predictive distribution of f*
function computefstar(model::FullBatchModel,X_test)
    n = size(X_test,1)
    ksize = model.nSamples
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = -(model.invK*(eye(ksize)-model.ζ*model.invK))
    end
    kstar = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    A = kstar*model.invK
    kstarstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    meanfstar = kstar*model.TopMatrixForPrediction
    covfstar = kstarstar + diag(A*(model.ζ*model.invK-eye(model.nSamples))*transpose(kstar))
    return meanfstar,covfstar
end
