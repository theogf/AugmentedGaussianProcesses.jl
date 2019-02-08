"Update all hyperparameters for the full batch GP models"
function  update_hyperparameters!(model::VGP)
    Jnn = kernelderivativematrix.([model.X],model.kernel)
    f_l,f_v = hyperparameter_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.kernel,fill(f_l,model.nPrior),Jnn,1:model.nPrior)
    grads_v = map(f_v,model.kernel,1:model.nPrior)
    apply_gradients_lengthscale!.(model.kernel,grads_l) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_variance!.(model.kernel,grads_v) #Send the derivative of the matrix to the specific gradient of the model
    model.inference.HyperParametersUpdated = true
end

"Update all hyperparameters for the full batch GP models"
function update_hyperparameters!(model::SVGP)
    matrix_derivatives =broadcast((kernel,Z)->
                    [kernelderivativematrix(Z,kernel), #Jmm
                     kernelderivativematrix(model.X[model.inference.MBIndices,:],Z,kernel), #Jnm
                     kernelderivativediagmatrix(model.X[model.inference.MBIndices,:],kernel)],#Jnn
                     model.kernel,model.Z)
    f_l,f_v = hyperparameter_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.kernel,fill(f_l,model.nPrior),matrix_derivatives,1:model.nPrior)
    grads_v = map(f_v,model.kernel,1:model.nPrior)
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
    A = (model.invKnn.*(model.Σ.+model.µ.*transpose.(model.μ)).-[I]).*model.invKnn
    if model.IndependentPriors
        return (function(Jnn,index)
                    return 0.5*sum(Jnn.*transpose(A[index]))
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*sum(model.Knn[index].*A[index]')
                end)
    else
        return (function(Jnn,index)
            return 0.5*sum(sum(Jnn.*transpose(A[i])) for i in 1:model.nLatent)
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel[1])*sum(sum(model.Knn[1].*transpose(A[i])) for i in 1:model.nLatent)
                end)
    end
end

# function hyerparameter_KL_gradient(A::AbstractMatrix,J::AbstractMatrix)
#
# end


"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::SVGP) where {T<:Real}
    A = (model.invKmm.*(model.Σ.+model.µ.*transpose.(model.μ)).-[I]).*model.invKmm
    ι = Matrix{Real}(undef,model.inference.nSamplesUsed,model.nFeature) #Empty container to save data allocation
    if model.IndependentPriors
        return (function(Jmm,Jnm,Jnn,index)
                    return 0.5*(hyperparameter_expec_gradient(model,ι,Jmm,Jnm,Jnn,index)
                            + opt_trace(Jmm,A[index]'))
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*(
                            dot(expec_Σ(model,index),model.K̃[index])
                            + opt_trace(model.Kmm[index],A[index]'))
                end)
    else
        return (function(Jmm,index)
                    return 0.5*sum(
                            hyperparameter_expec_gradient(model,ι,Jmm,Jnm,Jnn,i)
                            + opt_trace(Jmm,transpose(A[i])) for i in 1:model.nLatent)
                end,
                function(kernel,index)
                    return 0.5/getvariance(kernel)*sum(
                            dot(expec_Σ(model,i),model.K̃[1])
                            + opt_trace(model.Kmm[1],A[i]') for i in 1:model.nLatent)
                end)
    end
end


function hyperparameter_expec_gradient(model::SVGP,ι::AbstractArray,Jmm::AbstractMatrix,Jnm::AbstractMatrix,Jnn::AbstractVector,index::Integer)
    mul!(ι,(Jnm-model.κ[index]*Jmm),model.invKmm[index])
    Jnn .+= - opt_diag(ι,model.Knm[index]) - opt_diag(model.κ[index],Jnm)
    dμ = dot(model.inference.∇μE,ι*model.μ[index])
    dΣ = dot(model.inference.∇ΣE,Jnn+2.0*opt_diag((ι*model.Σ[index]),model.κ[index]))
    return model.inference.ρ*(dμ+dΣ)
end
