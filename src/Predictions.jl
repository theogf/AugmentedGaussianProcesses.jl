"""
File treating all the prediction functions
"""


#### Computation of predictions with and without variance using the probit and logit link ####
# function fstar(model::LinearModel,X_test,cov::Bool=true)
# end

function fstar(model::FullBatchModel,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = (model.invK*(eye(model.nSamples)-model.ζ*model.invK))
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        k_starstar = diagkernelmatrix(X_test,model.kernel)
        cov_fstar = k_starstar - sum((k_star*model.DownMatrixForPrediction).*k_star,2)
        return mean_fstar,cov_fstar
    end
end

function fstar(model::SparseModel,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.Kmm\model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = (model.Kmm\(eye(model.nFeatures)-model.ζ/model.Kmm))
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        k_starstar = diagkernelmatrix(X_test,model.kernel)
        cov_fstar = k_starstar - sum((k_star*model.DownMatrixForPrediction).*k_star,2)
        return mean_fstar,cov_fstar
    end
end

function fstar(model::MultiClass,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = broadcast(x->model.invK*x,model.μ)
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = broadcast(var->model.invK*(eye(model.nSamples)-var*model.invK),model.ζ)
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    mean_fstar = broadcast(x->k_star*x,model.TopMatrixForPrediction)
    if !covf
        return mean_fstar
    else
        k_starstar = diagkernelmatrix(X_test,model.kernel)
        cov_fstar = broadcast(x->(k_starstar - sum((k_star*x).*k_star,2)),model.DownMatrixForPrediction)
        return mean_fstar,cov_fstar
    end
end


function fstar(model::SparseMultiClass,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = broadcast((k,mu)->k\mu,model.Kmm,model.μ)
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = broadcast((var,Kmm)->(Kmm\(eye(model.nFeatures)-var/Kmm)),model.ζ,model.Kmm)
    end
    k_star = broadcast(points->kernelmatrix(X_test,points,model.kernel),model.inducingPoints)
    mean_fstar = k_star.*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        k_starstar = diagkernelmatrix(X_test,model.kernel)
        cov_fstar = broadcast((x,k)->(k_starstar - sum((k*x).*k,2)),model.DownMatrixForPrediction,k_star)
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
    n_test = size(X_test,1)
    @assert minimum(cov_f)>0  error("Covariance under 0")
    predic = zeros(n_test)
    for i in 1:n_test
        d= Normal(m_f[i],cov_f[i])
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
    return fstar(model,X_test)
end

#Return the mean and variance of the predictive distribution of f*
function regpredictproba(model::SparseModel,X_test)
    return fstar(model,X_test)
end

function multiclasspredict(model::MultiClass,X_test)
    n = size(X_test,1)
    m_f = fstar(model,X_test,covf=false)
    y = hcat(broadcast(x->logit.(x),m_f))
    predic = zeros(Int64,n)
    for i in 1:n
        predic[i] = findmax(broadcast(x->x[i],y))[2]
    end
    return model.class_mapping[predic]
end

function multiclasspredict(model::SparseMultiClass,X_test)
    n=size(X_test,1)
    m_f = fstar(model,X_test,covf=false)
    y = hcat(broadcast(x->logit.(x),m_f))
    predic = zeros(Int64,n)
    for i in 1:n
        predic[i] = findmax(broadcast(x->x[i],y))[2]
    end
    return model.class_mapping[predic]
end

function multiclasspredictproba(model::MultiClass,X_test)
end
