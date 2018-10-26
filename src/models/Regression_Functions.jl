#Specific functions of the Gaussian Process regression models

"Update the variational parameters of the full batch model"
function variational_updates!(model::GPRegression,iter::Integer)
    #Nothing to do here
end

"Update the variational parameters of the sparse model"
function variational_updates!(model::SparseGPRegression,iter::Integer)
    (grad_η_1,grad_η_2) = natural_gradient_Regression(model)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Update the variational parameters of the online model"
function variational_updates!(model::OnlineGPRegression,iter::Integer)
    (grad_η_1,grad_η_2) = natural_gradient_Regression(model)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

function natural_gradient_Regression(model::SparseGPRegression)
    grad_1 = model.StochCoeff.*(model.κ'*model.y[model.MBIndices])./getvalue(model.noise)
    grad_2 = -0.5*(model.StochCoeff*(model.κ')*model.κ./getvalue(model.noise)+model.invKmm)
    return (grad_1,grad_2)
end

function natural_gradient_Regression(model::OnlineGPRegression)
    grad_1 = model.StochCoeff*model.κ'*model.y./getvalue(model.noise)
    grad_2 = -0.5*(model.StochCoeff*(model.κ')*model.κ./getvalue(model.noise)+model.invKmm)
    return (grad_1,grad_2)
end


"ELBO function for the basic GP Regression"
function ELBO(model::GPRegression)
    return -ExpecLogLikelihood(model)
end

"ELBO function for the sparse variational GP Regression"
function ELBO(model::SparseGPRegression)
    ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    # model.StochCoeff = model.nSamples/model.nSamplesUsed
    # ELBO = -0.5*model.nSamples*(log(model.noise)+log(2*pi))
    # ELBO += -0.5*model.StochCoeff*sum((model.y[model.MBIndices] - model.κ*model.μ).^2)/model.noise
    # ELBO += -0.5*model.StochCoeff*sum(model.Ktilde)./model.noise
    # ELBO += -0.5*model.StochCoeff/model.noise*sum((model.κ*model.Σ).*model.κ)
    # ELBO += 0.5*(logdet(model.Σ)+logdet(model.invKmm))
    # ELBO += -0.5*(sum(model.invKmm.*transpose(model.Σ+model.μ*transpose(model.μ))))
    return -ELBO_v
end

function ExpecLogLikelihood(model::GPRegression)
    return -0.5*dot(model.y,model.invK*model.y)+0.5*logdet(model.invK)-0.5*model.nSamples*log(2*pi)
end

function ExpecLogLikelihood(model::SparseGPRegression)
    return -0.5*(model.nSamplesUsed*log(2π*getvalue(model.noise))
    + (sum((model.y[model.MBIndices]-model.κ*model.μ).^2)
    + sum(model.Ktilde)+sum((model.κ*model.Σ).*model.κ))/getvalue(model.noise))
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a Regression Model"
function hyperparameter_gradient_function(model::GPRegression)
    A = model.invK*(model.y*transpose(model.y))-Diagonal{Float64}(I,model.nSamples)
    return (function(Jmm)
                V = model.invK*Jmm
                return 0.5*sum(V.*transpose(A))
            end,
            function(kernel)
                return 0.5/getvariance(kernel)*tr(A)
            end,
            function()
                return 0.5*sum(model.invK.*transpose(A))
            end)
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a sparse regression Model"
function hyperparameter_gradient_function(model::SparseGPRegression)
    F2 = model.μ*transpose(model.μ) + model.Σ
    return (function(Jmm,Jnm,Jnn)
            ι = (Jnm-model.κ*Jmm)*model.invKmm
            Jtilde = Jnn - sum(ι.*model.Knm,dims=2) - sum(model.κ.*Jnm,dims=2)
            V = model.invKmm*Jmm
            return 0.5*(sum( (V*model.invKmm).*F2)
            - model.StochCoeff/getvalue(model.noise)*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V) - model.StochCoeff/getvalue(model.noise)*sum(Jtilde)
            + 2*model.StochCoeff/getvalue(model.noise)*dot(model.y[model.MBIndices],ι*model.μ))
            end,
            function(kernel)
                return 0.5/(getvariance(kernel))*(sum(model.invKmm.*F2)-model.m-model.StochCoeff/getvalue(model.noise)*sum(model.Ktilde))
            end,
            function()
                ι = -model.κ*model.invKmm
                Jtilde = ones(Float64,model.nSamplesUsed) - sum(ι.*model.Knm,dims=2)[:]
                V = model.invKmm
                return 0.5*(sum( (V*model.invKmm).*F2)
                - model.StochCoeff/getvalue(model.noise)*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V) - model.StochCoeff/getvalue(model.noise)*sum(Jtilde)
                + 2*model.StochCoeff/getvalue(model.noise)*dot(model.y[model.MBIndices],ι*model.μ))
            end)
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters"
function inducingpoints_gradient(model::SparseGPRegression)
        gradients_inducing_points = zero(model.inducingPoints)
        F2 = model.μ*transpose(model.μ) + model.Σ
        for i in 1:model.m #Iterate over the points
            Jnm,Jmm = computeIndPointsJ(model,i)
            for j in 1:model.nDim #iterate over the dimensions
                ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
                Jtilde = -sum(ι.*model.Knm,dims=2)-sum(model.κ.*Jnm[j,:,:],dims=2)
                V = model.invKmm*Jmm[j,:,:]
                gradients_inducing_points[i,j] = 0.5*(sum( (V*model.invKmm).*F2)
                - model.StochCoeff/getvalue(model.noise)*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V) - model.StochCoeff/getvalue(model.noise)*sum(Jtilde)
                + 2*model.StochCoeff/getvalue(model.noise)*dot(model.y[model.MBIndices],ι*model.μ))
            end
        end
        return gradients_inducing_points
end
