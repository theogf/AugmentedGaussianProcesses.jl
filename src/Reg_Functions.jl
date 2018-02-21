
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
    ELBO = -0.5*model.nSamples*(log(model.γ)+log(2*pi))
    ELBO += -0.5*model.StochCoeff*(model.y - model.κ*model.μ).^2/model.γ
    ELBO += -0.5*model.StochCoeff*sum(model.Ktilde)./model.γ
    ELBO += -0.5*model.StochCoeff/model.γ*sum((model.ζ*model.kappa).*model.kappa)
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
    B = model.μ*transpose(model.μ) + model.ζ
    Kmn = CreateKernelMatrix(model.inducingPoints,model.Kernel_function;X2=model.X[model.MBIndices,:])
    dim = size(model.X,2)
    #Update of both the coefficients and hyperparameters of the kernels
    gradients_kernel_param = zeros(model.nKernels)
    gradients_kernel_coeff = zeros(model.nKernels)
    for i in 1:model.nKernels
        #Compute the derivative of the kernel matrices given the kernel lengthscale
        Jnm_param,Jnn_param,Jmm_param = model.Kernels[i].coeff.*computeJ(model,model.Kernels[i].compute_deriv)
        ι_param = (Jnm_param-model.κ*Jmm_param)*model.invKmm
        Jtilde_param = Jnn_param - sum(ι_param.*(Kmn.'),2) - sum(model.κ.*Jnm_param,2)
        V_param = model.invKmm*Jmm_param
        gradients_kernel_param[i] = 0.5*(sum( (V_param*model.invKmm - model.StochCoeff/model.γ*(ι_param'*model.κ + model.κ'*ι_param)) .* transpose(B)) - trace(V_param) - model.StochCoeff/model.γ*sum(Jtilde_param)
         + 2*model.StochCoeff/model.γ*dot(model.y[model.MBIndices],ι_param*model.μ))
        #Compute the derivative of the kernel matrices given the kernel weight
        Jnm_coeff,Jnn_coeff,Jmm_coeff = computeJ(model,model.Kernels[i].compute)
        ι_coeff = (Jnm_coeff-model.κ*Jmm_coeff)*model.invKmm
        Jtilde_coeff = Jnn_coeff - sum(ι_coeff.*(Kmn.'),2) - sum(model.κ.*Jnm_coeff,2)
        V_coeff = model.invKmm*Jmm_coeff
        gradients_kernel_coeff[i] = 0.5*(sum(( V_coeff*model.invKmm - model.StochCoeff/model.γ*(ι_coeff'*model.κ + model.κ'*ι_coeff)) .* transpose(B)) -trace(V_coeff) - model.StochCoeff/model.γ*sum(Jtilde_coeff)
         + 2*model.StochCoeff/model.γ*dot(model.y[model.MBIndices],ι_coeff*model.μ))
    end
    if model.OptimizeInducingPoints
        gradients_inducing_points = zeros(model.m,dim)
        for i in 1:model.m #Iterate over the points
            Jnm,Jmm = computeIndPointsJ(model,i)
            for j in 1:dim #iterate over the dimensions
                ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
                Jtilde = -sum(ι.*(Kmn.'),2)-sum(model.κ.*Jnm[j,:,:],2)
                V = model.invKmm*Jmm[j,:,:]
                gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm-model.StochCoeff/model.γ*(ι'*model.κ+model.κ'*ι)).*transpose(B))-trace(V)-model.StochCoeff/model.γ*Jtilde
                 + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
            end
        end
        return gradients_kernel_param,gradients_kernel_coeff,gradients_inducing_points
    else
        return gradients_kernel_param,gradients_kernel_coeff
    end
end
