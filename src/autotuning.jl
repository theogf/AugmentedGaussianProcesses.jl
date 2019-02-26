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
    apply_gradients_lengthscale!.(model.kernel,grads_l)
    apply_gradients_variance!.(model.kernel,grads_v)
    model.inference.HyperParametersUpdated = true
end

function hyperparameter_KL_gradient(J::AbstractMatrix{T},A::AbstractMatrix{T}) where {T<:Real}
    return 0.5*opt_trace(J,A)
end


"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::VGP) where {T<:Real}
    A = (model.invKnn.*(model.Σ.+model.µ.*transpose.(model.μ)).-[I]).*model.invKnn
    if model.IndependentPriors
        return (function(Jnn,index)
                    return hyperparameter_KL_gradient(Jnn,A[index])
                end,
                function(kernel,index)
                    return 1.0/getvariance(kernel)*hyperparameter_KL_gradient(model.Knn[index],A[index])
                end)
    else
        return (function(Jnn,index)
            return sum(hyperparameter_KL_gradient.([Jnn],A))
                end,
                function(kernel,index)
                    return 1.0/getvariance(kernel[1])*sum(hyperparameter_KL_gradient.(model.Knn,A))
                end)
    end
end

"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::SVGP{<:Likelihood,<:Inference,T,<:Any}) where {T<:Real}
    A = ([I].-model.invKmm.*(model.Σ.+model.µ.*transpose.(model.μ))).*model.invKmm
    ι = Matrix{T}(undef,model.inference.nSamplesUsed,model.nFeature) #Empty container to save data allocation
    if model.IndependentPriors
        return (function(Jmm,Jnm,Jnn,index)
                    return hyperparameter_expec_gradient(model,ι,Jmm,Jnm,Jnn,index)
                            - hyperparameter_KL_gradient(Jmm,A[index])
                end,
                function(kernel,index)
                    return 1.0/getvariance(kernel)*(
                            - model.inference.ρ*dot(expec_Σ(model,index),model.K̃[index])
                            - hyperparameter_KL_gradient(model.Kmm[index],A[index]))
                end)
    else
        return (function(Jmm,Jnm,Jnn,index)
                    return  hyperparameter_expec_gradient(model,ι,Jmm,Jnm,Jnn)
                           - sum(hyperparameter_KL_gradient.([Jmm],A))
                end,
                function(kernel,index)
                    return 1.0/getvariance(kernel)*(model.inference.ρ*sum(
                            - dot(expec_Σ(model,i),model.K̃[1]) for i in 1:model.nLatent)
                            - sum(hyperparameter_KL_gradient.(model.Kmm,A)))
                end)
    end
end


function hyperparameter_expec_gradient(model::SVGP,ι::AbstractArray,Jmm::AbstractMatrix,Jnm::AbstractMatrix,Jnn::AbstractVector,index::Integer)
    mul!(ι,(Jnm-model.κ[index]*Jmm),model.invKmm[index])
    Jnn .-= opt_diag(ι,model.Knm[index]) + opt_diag(model.κ[index],Jnm)
    dμ = dot(expec_μ(model,index),ι*model.μ[index])
    dΣ = -dot(expec_Σ(model,index),Jnn+2.0*(opt_diag(ι*model.Σ[index],model.κ[index])+(ι*model.μ[index]).*(model.κ[index]*model.μ[index])))
    return model.inference.ρ*(dμ+dΣ)
end

function hyperparameter_expec_gradient(model::SVGP,ι::AbstractArray,Jmm::AbstractMatrix,Jnm::AbstractMatrix,Jnn::AbstractVector)
    mul!(ι,(Jnm-model.κ[1]*Jmm),model.invKmm[1])
    Jnn .-= opt_diag(ι,model.Knm[1]) + opt_diag(model.κ[1],Jnm)
    dμ = sum(dot(expec_μ(model,i),ι*model.μ[i]) for i in 1:model.nLatent)
    dΣ = -sum(dot(expec_Σ(model,i),Jnn+2.0*(opt_diag(ι*model.Σ[i],model.κ[1])+(ι*model.μ[i]).*(model.κ[1]*model.μ[i]))) for i in 1:model.nLatent)
    return model.inference.ρ*(dμ+dΣ)
end