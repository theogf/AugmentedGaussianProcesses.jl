
#Specific functions of the Gaussian Process regression models


"""
    ELBO function for the basic GP Regression
"""
function ELBO(model::GPRegression)
    return -0.5*dot(model.y,model.invK*model.y)+0.5logdet(model.invK)-model.nSamples*log(2*pi)
end

"""
    ELBO function for the sparse variational GP Regression
"""
function ELBO(model::SparseGPRegression)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO = -0.5*model.nSamples*(log(model.noise)+log(2*pi))
    ELBO += -0.5*model.StochCoeff*sum((model.y[model.MBIndices] - model.κ*model.μ).^2)/model.noise
    ELBO += -0.5*model.StochCoeff*sum(model.Ktilde)./model.noise
    ELBO += -0.5*model.StochCoeff/model.noise*sum((model.κ*model.ζ).*model.κ)
    ELBO += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
    ELBO += -0.5*(sum(model.invKmm.*transpose(model.ζ+model.μ*transpose(model.μ))))
    return -ELBO
end


function variablesUpdate_Regression!(model::GPRegression,iter)
    #Nothing to do here
end

"""
    Update the variational parameters of the model
"""
function variablesUpdate_Regression!(model::SparseGPRegression,iter)
    (grad_η_1,grad_η_2) = naturalGradientELBO_Regression(model.y[model.MBIndices],model.κ,model.noise,stoch_coeff=model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function naturalGradientELBO_Regression(y,κ,noise;stoch_coeff=1.0)
    grad_1 = stoch_coeff*κ'*y./noise
    grad_2 = -0.5*(stoch_coeff*(κ')*κ./noise)
    return (grad_1,grad_2)
end


function hyper_parameter_gradient_function(model::GPRegression)
    A = model.invK*(model.y*transpose(model.y))-eye(model.nSamples)
    return function(Js)
                V = model.invK*Js[1]
                return 0.5*sum(V_param.*transpose(A))
            end
end

function hyperparameter_gradient_function(model::SparseGPRegression)
    B = model.μ*transpose(model.μ) + model.ζ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    return function(Js)
            Jmm = Js[1]; Jnm = Js[2]; Jnn = Js[3];
            ι = (Jnm-model.κ*Jmm)*model.invKmm
            Jtilde = Jnn - sum(ι.*(Kmn.'),2) - sum(model.κ.*Jnm,2)
            V = model.invKmm*Jmm
            return 0.5*(sum( (V*model.invKmm - model.StochCoeff/model.noise*(ι'*model.κ + model.κ'*ι)) .* transpose(B)) - trace(V) - model.StochCoeff/model.noise*sum(Jtilde)
             + 2*model.StochCoeff/model.noise*dot(model.y[model.MBIndices],ι*model.μ))
        end
end

function inducingpoints_gradient(model::SparseGPRegression)
        gradients_inducing_points = zeros(model.inducingPoints)
        for i in 1:model.m #Iterate over the points
            Jnm,Jmm = computeIndPointsJ(model,i)
            for j in 1:dim #iterate over the dimensions
                ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
                Jtilde = -sum(ι.*(Kmn.'),2)-sum(model.κ.*Jnm[j,:,:],2)
                V = model.invKmm*Jmm[j,:,:]
                gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm-model.StochCoeff/model.noise*(ι'*model.κ+model.κ'*ι)).*transpose(B))-trace(V)-model.StochCoeff/model.noise*Jtilde
                 + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
            end
        end
        return gradients_inducing_points
end
