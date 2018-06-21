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
        model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = (model.invKmm*(eye(model.nFeatures)-model.ζ*model.invKmm))
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
        if model.KInducingPoints
            model.TopMatrixForPrediction = broadcast((k,mu)->k\mu,model.Kmm,model.μ)
        else
            model.TopMatrixForPrediction = broadcast((mu)->model.Kmm[1]\mu,model.μ)
        end
    end
    if covf && model.DownMatrixForPrediction == 0
        if model.KInducingPoints
            model.DownMatrixForPrediction = broadcast((var,Kmm)->(Kmm\(eye(model.nFeatures)-var/Kmm)),model.ζ,model.Kmm)
        else
            model.DownMatrixForPrediction = broadcast((var)->(model.Kmm[1]\(eye(model.nFeatures)-var/model.Kmm[1])),model.ζ)
        end
    end
    if model.KInducingPoints
        k_star = broadcast(points->kernelmatrix(X_test,points,model.kernel),model.inducingPoints)
    else
        k_star = [kernelmatrix(X_test,model.inducingPoints[1],model.kernel)]
    end
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

function multiclasspredict(model::MultiClass,X_test,all_class=false)
    n = size(X_test,1)
    m_f = fstar(model,X_test,covf=false)
    σ = hcat(logit.(m_f)...)
    σ = [σ[i,:] for i in 1:n]
    normsig = sum.(σ)
    y = mod_soft_max.(σ,normsig)
    if all_class
        return y
    end
    predic = zeros(Int64,n)
    value = zeros(Float64,n)
    for i in 1:n
        res = findmax(y[i]);
        predic[i]=res[2];
        value[i]=res[1]
    end
    # broadcast((x,pred,val)->begin ;end,y,predic,value)
    return model.class_mapping[predic],value
end

function multiclasspredict(model::SparseMultiClass,X_test,all_class=false)
    n=size(X_test,1)
    m_f = fstar(model,X_test,covf=false)
    σ = hcat(logit.(m_f)...)
    σ = [σ[i,:] for i in 1:n]
    normsig = sum.(σ)
    y = mod_soft_max.(σ,normsig)
    if all_class
        return y
    end
    predic = zeros(Int64,n)
    value = zeros(Float64,n)
    for i in 1:n
        res = findmax(y[i]);
        predic[i]=res[2];
        value[i]=res[1]
    end
    return model.class_mapping[predic],value
end



function multiclasspredictproba(model::MultiClass,X_test)
    n = size(X_test,1)
    m_f,cov_f = fstar(model,X_test)
    σ = hcat(logit.(m_f)...)
    σ = [σ[i,:] for i in 1:n]
    cov_f = hcat(cov_f...)
    cov_f = [cov_f[i,:] for i in 1:n]
    normsig = sum.(σ)
    h = mod_soft_max.(σ,normsig)
    grad_h = grad_mod_soft_max.(σ,normsig)
    hess_h = hessian_mod_soft_max.(σ,normsig)
    m_predic = h.+0.5*broadcast((hess,cov)->(hess*cov),hess_h,cov_f)
    cov_predic = broadcast((grad,hess,cov)->(grad.^2*cov-0.25*hess.^2*(cov.^2)),grad_h,hess_h,cov_f)
    return m_predic,cov_predic
end
function multiclasspredictprobamcmc(model::MultiClass,X_test,NSamples=100)
    n = size(X_test,1)
    m_f,cov_f = fstar(model,X_test)
    m_f = hcat(m_f...)
    m_f = [m_f[i,:] for i in 1:n]
    cov_f = hcat(cov_f...)
    cov_f = [cov_f[i,:] for i in 1:n]
    stack_preds = Array{Array{Any,1},1}(n);
    m_pred_mc = Array{Array{Float64,1},1}(n)
    sig_pred_mc = Array{Array{Float64,1},1}(n)
    for i in 1:n
        preds = []
        if i%100 == 0
            println("$i/$n points predicted with sampling ($NSamples samples)")
        end
        for samp in 1:NSamples
            samp = logit.(broadcast((m,cov)->rand(Normal(m,cov)),m_f[i],cov_f[i]))
            norm_sig = sum(samp)
            push!(preds,mod_soft_max(samp,norm_sig))
        end
        m_pred_mc[i]=mean(preds)
        sig_pred_mc[i]=cov.([broadcast(x->x[j],preds) for j in 1:model.K])
    end
    return m_pred_mc,sig_pred_mc
end

function mod_soft_max(σ,normsig)
    return σ./normsig
end

function grad_mod_soft_max(σ,normsig)
    short_sum = normsig-σ
    norm_square = normsig^2
    base_grad = (σ-(σ.^2))./norm_square
    n = size(σ,1)
    grad = zeros(n,n)
    for i in 1:n
        for j in 1:n
            if i==j
                grad[i,i] = short_sum[i]*base_grad[i]
            else
                grad[i,j] = -σ[i]*base_grad[j]
            end
        end
    end
    return grad
end

function hessian_mod_soft_max(σ,normsig)
    short_sum = normsig-σ
    norm_square = normsig^2
    norm_cube = normsig^3
    base_grad = (σ-σ.^2).*((1.0-2.0.*σ).*norm_square-2.0.*(σ-σ.^2)*normsig)./norm_cube
    n = size(σ,1)
    grad = zeros(n,n)
    for i in 1:n
        for j in 1:n
            if i==j
                grad[i,i] = short_sum[i]*base_grad[i]
            else
                grad[i,j] = -σ[i]*base_grad[j]
            end
        end
    end
    return grad
end
