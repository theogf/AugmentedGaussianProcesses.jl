


function ELBO(model::GPRegression)
    return -0.5*dot(model.y,model.invK*model.y)+0.5logdet(model.invK)-model.nSamples*log(2*pi)
end

function ELBO(model::SparseGPRegression)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO = -0.5*model.nSamples*(log(model.γ)+log(2*pi))
    ELBO += -model.StochCoeff*((model.y - model.κ*model.μ)./model.γ).^2
    ELBO += -0.5*model.StochCoeff*model.Ktilde./model.γ
    ELBO += -0.5*model.StochCoeff/model.γ*sum(model.kappa.*model.kappa)
    ELBO += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
    ELBO += -0.5*(sum(model.invKmm.*transpose(model.ζ+model.μ*transpose(model.μ))))
    return -ELBO
end


function variablesUpdate_Regression!(model::GPRegression,iter)
    #Nothing to do here
end


function variablesUpdate_Regression!(model::SparseGPRegression,iter)
    (grad_η_1,grad_η_2) = naturalGradientELBO_Regression(model.y[model.MBIndices],model.κ,model.γ,stoch_coeff=model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function naturalGradientELBO_Regression(y,κ,γ;stoch_coeff=1.0)
    grad_1 = stoch_coeff*κ'*y./γ
    grad_2 = -0.5*(stoch_coeff*(κ')*κ./γ)
    return (grad_1,grad_2)
end


function computeHyperParametersGradients(model::GPRegression,iter::Integer)
    A = model.invK*(model.y*transpose(model.y))-eye(model.nSamples)
    #Update of both the coefficients and hyperparameters of the kernels
    gradients_kernel_param = zeros(model.nKernels)
    gradients_kernel_coeff = zeros(model.nKernels)
    for i in 1:model.nKernels
        V_param = model.invK*model.Kernels[i].coeff*computeJ(model,model.Kernels[i].compute_deriv)
        V_coeff = model.invK*computeJ(model,model.Kernels[i].compute)
        gradients_kernel_param[i] = 0.5*sum(V_param.*transpose(A))
        gradients_kernel_coeff[i] = 0.5*sum(V_coeff.*transpose(A))
    end
    return gradients_kernel_param,gradients_kernel_coeff
end

function computeHyperParametersGradients(model::SparseGPRegression,iter::Integer)
