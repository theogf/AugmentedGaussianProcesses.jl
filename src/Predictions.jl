#File treating all the prediction functions

"""
Compute the mean of the predicted latent distribution of f on X_test for full GP models
Return also the variance if `covf=true`
"""
function fstar(model::FullBatchModel,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = (model.invK*(Diagonal{Float64}(I,model.nSamples)-model.ζ*model.invK))
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        cov_fstar = diagkernelmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
        return mean_fstar,cov_fstar
    end
end
"""
Compute the mean of the predicted latent distribution of f on X_test for sparse GP models
Return also the variance if `covf=true`
"""
function fstar(model::SparseModel,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = (model.invKmm*(Diagonal{Float64}(I,model.nFeatures)-model.ζ*model.invKmm))
    end
    k_star = kernelmatrix(X_test,model.inducingPoints,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        cov_fstar = diagkernelmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
        return mean_fstar,cov_fstar
    end
end

"""
Compute the mean of the predicted latent distribution of f on X_test for online GP models
Return also the variance if `covf=true`
"""
function fstar(model::OnlineGPModel,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = (model.invKmm*(Diagonal{Float64}(I,model.nFeatures)-model.ζ*model.invKmm))
    end
    k_star = kernelmatrix(X_test,model.kmeansalg.centers,model.kernel)
    mean_fstar = k_star*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        cov_fstar = diagkernelmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
        return mean_fstar,cov_fstar
    end
end

"""
Compute the mean of the predicted latent distribution of f on X_test for GP regression
Return also the variance if `covf=true`
"""
function fstar(model::GPRegression,X_test;covf::Bool=true)
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
        cov_fstar = diagkernelmatrix(X_test,model.kernel) .- sum((k_star*model.DownMatrixForPrediction).*k_star,dims=2)[:]
        return mean_fstar,cov_fstar
    end
end

"""
Compute the mean of the predicted latent distribution of f on X_test for Multiclass GP models
Return also the variance if `covf=true`
"""
function fstar(model::MultiClass,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = broadcast((mu,invK)->invK*mu,model.μ,model.invK)
    end
    if covf && model.DownMatrixForPrediction == 0
      model.DownMatrixForPrediction = broadcast((var,invK)->invK*(Diagonal{Float64}(I,model.nSamples)-var*invK),model.ζ,model.invK)
    end
    if model.IndependentGPs
        k_star = [kernelmatrix(X_test,model.X,model.kernel[i]) for i in 1:model.K]
    else
        k_star = [kernelmatrix(X_test,model.X,model.kernel[1])]
    end
    mean_fstar = broadcast((k_s,m)->k_s*m,k_star,model.TopMatrixForPrediction)
    if !covf
        return mean_fstar
    else
        if model.IndependentGPs
            k_starstar = [diagkernelmatrix(X_test,model.kernel[i]) for i in 1:model.K]
        else
            k_starstar = [diagkernelmatrix(X_test,model.kernel[1])]
        end
        cov_fstar = broadcast((k_ss,k_s,x)->(k_ss .- sum((k_s*x).*k_s,dims=2)[:]),k_starstar,k_star,model.DownMatrixForPrediction)
        return mean_fstar,cov_fstar
    end
end

"""
Compute the mean of the predicted latent distribution of f on X_test for multiclass sparse GP models
Return also the variance if `covf=true`
"""
function fstar(model::SparseMultiClass,X_test;covf::Bool=true)
    if model.TopMatrixForPrediction == 0
        if model.IndependentGPs
            model.TopMatrixForPrediction = broadcast((k,mu)->k\mu,model.Kmm,model.μ)
        else
            model.TopMatrixForPrediction = broadcast((mu)->model.Kmm[1]\mu,model.μ)
        end
    end
    if covf && model.DownMatrixForPrediction == 0
        if model.IndependentGPs
            model.DownMatrixForPrediction = broadcast((var,Kmm)->(Kmm\(Diagonal{Float64}(I,model.nFeatures)-var/Kmm)),model.ζ,model.Kmm)
        else
            model.DownMatrixForPrediction = broadcast((var)->(model.Kmm[1]\(Diagonal{Float64}(I,model.nFeatures)-var/model.Kmm[1])),model.ζ)
        end
    end
    if model.IndependentGPs
        k_star = broadcast((points,kernel)->kernelmatrix(X_test,points,kernel),model.inducingPoints,model.kernel)
    else
        k_star = [kernelmatrix(X_test,model.inducingPoints[1],model.kernel[1])]
    end
    mean_fstar = k_star.*model.TopMatrixForPrediction
    if !covf
        return mean_fstar
    else
        if model.IndependentGPs
            k_starstar = diagkernelmatrix.(X_test,model.kernel)
        else
            k_starstar = [diagkernelmatrix(X_test,model.kernel[1])]
        end
        cov_fstar = broadcast((x,ks,kss)->(kss .- sum((ks*x).*ks,dims=2)),model.DownMatrixForPrediction,k_star,k_starstar)
        return mean_fstar,cov_fstar
    end
end

"Return the predicted class {-1,1} with a linear model via the probit link"
function probitpredict(model::LinearModel,X_test)
    return sign.((model.Intercept ? [ones(Float64,size(X_test,1)) X_test]*model.μ : X_test*model.μ).-0.5)
end

"Return the predicted class {-1,1} with a GP model via the probit link"
function probitpredict(model::GPModel,X_test)
    return sign.(fstar(model,X_test,covf=false).-0.5)
end

"Return the mean of likelihood p(y*=1|X,x*) via the probit link with a linear model"
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

"Return the mean of likelihood p(y*=1|X,x*) via the probit link with a GP model"
function probitpredictproba(model::GPModel,X_test)
    m_f,cov_f = fstar(model,X_test,covf=true)
    return broadcast((m,c)->cdf(Normal(),m/(c+1)),m_f,cov_f)
end

"Return logit(x)"
function logit(x)
    return 1.0./(1.0.+exp.(-x))
end


"Return the predicted class {-1,1} with a GP model via the logit link"
function logitpredict(model::GPModel,X_test)
    return sign.(fstar(model,X_test,covf=false).-0.5)
end

"Return the mean of likelihood p(y*=1|X,x*) via the logit link with a GP model"
function logitpredictproba(model::GPModel,X_test)
    m_f,cov_f = fstar(model,X_test,covf=true)
    n_test = size(X_test,1)
    @assert minimum(cov_f)>0  error("Covariance under 0")
    predic = zeros(Float64,n_test)
    for i in 1:n_test
        d = Normal(m_f[i],sqrt(cov_f[i]))
        predic[i] = quadgk(x->logit(x)*pdf(d,x),-Inf,Inf)[1]
        # predic[i] = quadgk(x->logit(x)*pdf(d,x),m_f[i]-10*sqrt(cov_f[i]),m_f[i]+10*sqrt(cov_f[i]))[1]
        # err += abs(v-predic[i])
    end
    return predic
end

"""Return the mean of the predictive distribution of f"""
function regpredict(model::GPRegression,X_test)
    if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invK*model.y
    end
    k_star = kernelmatrix(X_test,model.X,model.kernel)
    return k_star*model.TopMatrixForPrediction
end

"""Return the mean of the predictive distribution of f"""
function regpredict(model::GPModel,X_test)
    return fstar(model,X_test,covf=false)
end

"""Return the mean and variance of the predictive distribution of f"""
function regpredictproba(model::GPModel,X_test)
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


"Return the modified softmax likelihood given the array of 'σ' and their sum (can be given via sumsig)"
function mod_soft_max(σ::Array{Float64,1},sumsig::Float64=0.0)
    return sumsig == 0 ? σ./(sum(σ)) : σ./sumsig
end


"Return the gradient of the modified softmax likelihood given 'σ' and their sum (can be given via sumsig)"
function grad_mod_soft_max(σ::Array{Float64,1},sumsig::Float64=0.0)
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
function hessian_mod_soft_max(σ::Array{Float64,1},summsig::Float64=0.0)
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
