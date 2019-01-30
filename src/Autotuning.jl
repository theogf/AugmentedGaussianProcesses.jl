"Update all hyperparameters for the full batch GP models"
function  update_hyperparameters!(model::VGP)
    Jnn = kernelderivativematrix.([model.X],model.kernel)
    f_l,f_v = hyperparameter_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.kernel,[f_l],Jnn,1:model.nPrior)
    grads_v = map(f_v,model.kernel,1:model.nPrior)
    apply_gradients_lengthscale!.(model.kernel,grads_l) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_variance!.(model.kernel,grads_v) #Send the derivative of the matrix to the specific gradient of the model
    model.inference.HyperParametersUpdated = true
end

"Update all hyperparameters for the full batch GP models"
function update_hyperparameters!(model::SVGP)
    matrix_derivatives =broadcast((kernel,Z)->
                    [kernelderivativematrix(Z,kernel), #Jmm
                     kernelderivativematrix(model.X[model.MBIndices,:],Z,kernel), #Jnm
                     kernelderivativediagmatrix(model.X[model.MBIndices,:],kernel)],#Jnn
                     model.kernel,model.Z)
    f_l,f_v = hyperparameter_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.kernel,f_l,matrix_derivatives)
    grads_v = map(f_v,model.kernel)
    if model.OptimizeInducingPoints
        Z_gradients = inducingpoints_gradient(model) #Compute the gradient given the inducing points location
        model.Z += GradDescent.update(model.optimizer,Z_gradients) #Apply the gradients on the location
    end
    apply_gradients_lengthscale!(model.kernel,grads_l)
    apply_gradients_variance!(model.kernel,grads_v)
    model.HyperParametersUpdated = true
end



"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::VGP) where {T<:Real}
    A = (model.invKnn.*(model.Σ.+model.µ.*transpose.(model.μ)).-I).*model.invKnn
    if model.IndependentPriors
        return (function(Jnn,index)
                    return 0.5*sum(J.*transpose(A[index]))
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*sum(model.Knn[index].*A[index]')
                end)
    else
        return (function(J,index)
            return 0.5*sum(sum(J.*transpose(A[i])) for i in 1:model.nLatent)
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*sum(sum(model.Knn[1].*transpose(A[i])) for i in 1:model.nLatent)
                end)
    end
end

# function hyerparameter_KL_gradient(A::AbstractMatrix,J::AbstractMatrix)
#
# end


"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::SVGP) where {T<:Real}
    A = (model.invKmm.*(model.Σ.+model.µ.*transpose.(model.μ)).-I).*model.invKmm
    if model.IndependentPriors
        return (function(Jmm,Jnm,Jnn,index)
                    grad_KL =  0.5*sum(Jmm.*transpose(A[index]))
                    grad_Expec = hyerparameter_expec_gradient(model)
                    return grad_KL + grad_Expec #TODO soething like this
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*sum(model.Knn[index].*A[index]')
                end)
    else
        return (function(J,index)
            return 0.5*sum(sum(J.*transpose(A[i])) for i in 1:model.nLatent)
                end,
                function(kernel)
                    return 0.5/getvariance(kernel)*sum(sum(model.Knn[1].*transpose(A[i])) for i in 1:model.nLatent)
                end)
    end
end
