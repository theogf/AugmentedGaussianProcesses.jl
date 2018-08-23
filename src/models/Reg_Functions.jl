#Specific functions of the Gaussian Process regression models

"Update the variational parameters of the full batch model"
function variational_updates!(model::GPRegression,iter::Integer)
    #Nothing to do here
end

"Update the variational parameters of the sparse model"
function variational_updates!(model::SparseGPRegression,iter::Integer)
    (grad_η_1,grad_η_2) = natural_gradient_Regression(model.y[model.MBIndices],model.κ,model.noise,stoch_coeff=model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Update the variational parameters of the online model"
function variational_updates!(model::OnlineGPRegression,iter::Integer)
    (grad_η_1,grad_η_2) = natural_gradient_Regression(model.y[model.MBIndices],model.κ,model.noise,stoch_coeff=model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

function natural_gradient_Regression(y::Vector{Float64},κ::Matrix{Float64},noise::Float64;stoch_coeff::Float64=1.0)
    grad_1 = stoch_coeff*κ'*y./noise
    grad_2 = -0.5*(stoch_coeff*(κ')*κ./noise)
    return (grad_1,grad_2)
end


"ELBO function for the basic GP Regression"
function ELBO(model::GPRegression)
    return -0.5*dot(model.y,model.invK*model.y)+0.5logdet(model.invK)-model.nSamples*log(2*pi)
end

"ELBO function for the sparse variational GP Regression"
function ELBO(model::SparseGPRegression)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO = -0.5*model.nSamples*(log(model.noise)+log(2*pi))
    ELBO += -0.5*model.StochCoeff*sum((model.y[model.MBIndices] - model.κ*model.μ).^2)/model.noise
    ELBO += -0.5*model.StochCoeff*sum(model.Ktilde)./model.noise
    ELBO += -0.5*model.StochCoeff/model.noise*sum((model.κ*model.Σ).*model.κ)
    ELBO += 0.5*(logdet(model.Σ)+logdet(model.invKmm))
    ELBO += -0.5*(sum(model.invKmm.*transpose(model.Σ+model.μ*transpose(model.μ))))
    return -ELBO
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a Regression Model"
function hyper_parameter_gradient_function(model::GPRegression)
    A = model.invK*(model.y*transpose(model.y))-Diagonal{Float64}(I,model.nSamples)
    return function(Js,iter)
                V = model.invK*Js[1]
                return 0.5*sum(V_param.*transpose(A))
            end
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a sparse regression Model"
function hyperparameter_gradient_function(model::SparseGPRegression)
    B = model.μ*transpose(model.μ) + model.Σ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    return function(Js,iter)
            Jmm = Js[1]; Jnm = Js[2]; Jnn = Js[3];
            ι = (Jnm-model.κ*Jmm)*model.invKmm
            Jtilde = Jnn - sum(ι.*transpose(Kmn),dims=2) - sum(model.κ.*Jnm,dims=2)
            V = model.invKmm*Jmm
            return 0.5*(sum( (V*model.invKmm - model.StochCoeff/model.noise*(ι'*model.κ + model.κ'*ι)) .* transpose(B)) - tr(V) - model.StochCoeff/model.noise*sum(Jtilde)
             + 2*model.StochCoeff/model.noise*dot(model.y[model.MBIndices],ι*model.μ))
        end
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters"
function inducingpoints_gradient(model::SparseGPRegression)
        gradients_inducing_points = zeros(model.inducingPoints)
        for i in 1:model.m #Iterate over the points
            Jnm,Jmm = computeIndPointsJ(model,i)
            for j in 1:dim #iterate over the dimensions
                ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
                Jtilde = -sum(ι.*transpose(Kmn),dims=2)-sum(model.κ.*Jnm[j,:,:],dims=2)
                V = model.invKmm*Jmm[j,:,:]
                gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm-model.StochCoeff/model.noise*(ι'*model.κ+model.κ'*ι)).*transpose(B))-tr(V)-model.StochCoeff/model.noise*Jtilde
                 + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
            end
        end
        return gradients_inducing_points
end
