#Specific functions of the Gaussian Process regression models

"""Local updates for regression (empty)"""
function local_update!(model::BatchGPRegression)
end

"""Local updates for regression (empty)"""
function local_update!(model::SparseGPRegression)
    model.gnoise = 1.0/model.nSamplesUsed * ( dot(model.y[model.MBIndices],model.y[model.MBIndices])
    - 2.0*dot(model.y[model.MBIndices],model.κ*model.μ)
    + sum((model.κ'*model.κ).*(model.μ*model.μ'+model.Σ)) + sum(model.Ktilde) )
end

"""Local updates for regression (empty)"""
function local_update!(model::OnlineGPRegression)
end

"Update the variational parameters of the full batch model for GP regression (empty)"
function variational_updates!(model::BatchGPRegression,iter::Integer)
    #Nothing to do here
end

"""Natural gradient computation for the sparse case"""
function natural_gradient(model::SparseGPRegression)
    grad_1 = model.StochCoeff.*(model.κ'*model.y[model.MBIndices])./model.gnoise
    grad_2 = Symmetric(-0.5*(model.StochCoeff*(model.κ')*model.κ./model.gnoise+model.invKmm))
    return (grad_1,grad_2)
end


function natural_gradient(model::OnlineGPRegression)
    grad_1 = model.StochCoeff*model.κ'*model.y./model.gnoise
    grad_2 = -0.5*(model.StochCoeff*(model.κ')*model.κ./model.gnoise+model.invKmm)
    return (grad_1,grad_2)
end


"""ELBO function for the basic GP Regression"""
function ELBO(model::BatchGPRegression)
    return -ExpecLogLikelihood(model)
end

"""ELBO function for the sparse variational GP Regression"""
function ELBO(model::SparseGPRegression)
    ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    return -ELBO_v
end

"""Return the expectation of the loglikelihood"""
function ExpecLogLikelihood(model::BatchGPRegression)
    return -0.5*dot(model.y,model.invK*model.y)+0.5*logdet(model.invK)-0.5*model.nSamples*log(2π)
end

"""Return the expectation of the loglikelihood for the sparse model"""
function ExpecLogLikelihood(model::SparseGPRegression)
    return -0.5*(model.nSamplesUsed*log(2π*model.gnoise)
    + (sum((model.y[model.MBIndices]-model.κ*model.μ).^2)
    + sum(model.Ktilde)+sum((model.κ*model.Σ).*model.κ))/model.gnoise)
end

"""Return functions computing the gradients of the ELBO given the kernel hyperparameters for a Regression Model"""
function hyperparameter_gradient_function(model::BatchGPRegression{T}) where {T<:Real}
    A = model.invK*(model.y*transpose(model.y))-I
    return (function(Jmm)
                V = model.invK*Jmm
                return 0.5*sum(V.*transpose(A))
            end,
            function(kernel)
                return 0.5/getvariance(kernel)*tr(A)
            end)
end

"Return functions computing the gradients of the ELBO given the kernel hyperparameters for a sparse regression Model"
function hyperparameter_gradient_function(model::SparseGPRegression{T}) where {T<:Real}
    F2 = model.μ*transpose(model.μ) + model.Σ
    return (function(Jmm,Jnm,Jnn)
            ι = (Jnm-model.κ*Jmm)*model.invKmm
            Jtilde = Jnn - sum(ι.*model.Knm,dims=2) - sum(model.κ.*Jnm,dims=2)
            V = model.invKmm*Jmm
            return 0.5*(sum( (V*model.invKmm).*F2)
            - model.StochCoeff/model.gnoise*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V)
            - model.StochCoeff/model.gnoise*sum(Jtilde)
            + 2*model.StochCoeff/model.gnoise*dot(model.y[model.MBIndices],ι*model.μ))
            end,
            function(kernel)
                return 0.5/(getvariance(kernel))*(sum(model.invKmm.*F2)-model.m-model.StochCoeff/model.gnoise*sum(model.Ktilde))
            end,
            function()
                ι = -model.κ*model.invKmm
                Jtilde = ones(T,model.nSamplesUsed) - sum(ι.*model.Knm,dims=2)[:]
                V = model.invKmm
                return 0.5*(sum( (V*model.invKmm).*F2)
                - model.StochCoeff/model.gnoise*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V)
                - model.StochCoeff/model.gnoise*sum(Jtilde)
                + 2*model.StochCoeff/model.gnoise*dot(model.y[model.MBIndices],ι*model.μ))
            end)
end

"""Return a function computing the gradient of the ELBO given the inducing point locations"""
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
                - model.StochCoeff/model.gnoise*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V) - model.StochCoeff/model.gnoise*sum(Jtilde)
                + 2*model.StochCoeff/model.gnoise*dot(model.y[model.MBIndices],ι*model.μ))
            end
        end
        return gradients_inducing_points
end
