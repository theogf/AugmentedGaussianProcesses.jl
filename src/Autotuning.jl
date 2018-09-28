

"Update all hyperparameters for the linear models"
function updateHyperParameters!(model::LinearModel,iter::Integer)
    grad_noise = 0.5*((tr(model.Σ)+norm(model.μ))/(model.noise^2.0)-model.nFeatures/model.noise);
    model.noise += GradDescent.update(model.optimizers[1],grad_noise)
    model.HyperParametersUpdated = true
end

"Update all hyperparameters for the full batch GP models"
function updateHyperParameters!(model::FullBatchModel)
    Jnn = derivativekernelmatrix(model.kernel,model.X) #Compute all the derivatives of the matrix Knn given the kernel parameters
    apply_gradients!(model.kernel, #Send the derivative of the matrix to the specific gradient of the model
                    compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jnn]))
    model.HyperParametersUpdated = true
end

"Update all hyperparameters for the full batch GP models"
function updateHyperParameters!(model::SparseModel)
    Jmm = derivativekernelmatrix(model.kernel,model.inducingPoints) #Compute all the derivatives of the matrix Kmm given the kernel
    Jnm = derivativekernelmatrix(model.kernel,model.X[model.MBIndices,:],model.inducingPoints) #Compute all the derivative of the matrix Knm given the kernel
    Jnn = derivativediagkernelmatrix(model.kernel,model.X[model.MBIndices,:]) #Compute all the derivatives of the diagonal matrix Knn given the kernel
    grads = compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),true,[Jmm,Jnm,Jnn],1,1)
    apply_gradients!(model.kernel,grads)
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model) #Compute the gradient given the inducing points location
        model.inducingPoints -= GradDescent.update(model.optimizer,inducingpoints_gradients) #Apply the gradients on the location
    end
    model.HyperParametersUpdated = true
end


"Update all hyperparameters for the multiclass GP models"
function updateHyperParameters!(model::MultiClass)
    if model.IndependentGPs
        Jnn = [[derivativekernelmatrix(model.kernel[i],model.X)] for i in model.KIndices]
        grads = compute_hyperparameter_gradient.(model.kernel[model.KIndices],hyperparameter_gradient_function(model),true,Jnn,model.KIndices,1:model.nClassesUsed)
        # println(grads)
        apply_gradients!.(model.kernel[model.KIndices],grads)#compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn]))
    else
        Jnn = [[derivativekernelmatrix(model.kernel[1],model.X)]]
        grads = compute_hyperparameter_gradient.(model.kernel,hyperparameter_gradient_function(model),true,Jnn,1,1)
        apply_gradients!(model.kernel[1],grads[1])#compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn]))
    end
    model.HyperParametersUpdated = true
end


function updateHyperParameters!(model::SparseMultiClass)
    if model.IndependentGPs
        # matrix_derivatives = [[kernelderivativematrix(model.inducingPoints[kiter],model.kernel[kiter]),
        # kernelderivativematrix(model.X[model.MBIndices,:],model.inducingPoints[kiter],model.kernel[kiter]),
        # kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[kiter])] for kiter in model.KIndices]
        (f_l,f_v) = hyperparameter_gradient_function(model)
        # grads = [compute_hyperparameter_gradient(model.kernel[kiter],f,true,
        #                 matrix_derivatives[iter],kiter,iter) for (iter,kiter) in enumerate(model.KIndices)]
        # grads = [compute_hyperparameter_gradient(model.kernel[kiter],f,true,
                            # [kernelderivativematrix_K(model.inducingPoints[kiter],model.Kmm[kiter],model.kernel[kiter]),
                            # kernelderivativematrix_K(model.X[model.MBIndices,:],model.inducingPoints[kiter],model.Knm[kiter],model.kernel[kiter]),
                            # kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[kiter])],kiter,iter) for (iter,kiter) in enumerate(model.KIndices)]
        grads_l = map(compute_hyperparameter_gradient,model.kernel[model.KIndices],[f_l for _ in 1:model.nClassesUsed],trues(model.nClassesUsed),[[kernelderivativematrix(model.inducingPoints[kiter],model.kernel[kiter]),
        kernelderivativematrix(model.X[model.MBIndices,:],model.inducingPoints[kiter],model.kernel[kiter]),
        kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[kiter])] for kiter in model.KIndices],model.KIndices,1:model.nClassesUsed)
        # println([getvalue(k.variance) for k in model.kernel])
        grads_variance = map(f_v,model.kernel[model.KIndices],model.KIndices,1:model.nClassesUsed)
        # println("Variances grad :", grads_variance)
        apply_gradients_lengthscale!.(model.kernel[model.KIndices],grads_l)
        apply_gradients_variance!.(model.kernel[model.KIndices],grads_variance)
        # setvariance(model)
    else
        matrix_derivatives = [kernelderivativematrix(model.inducingPoints[1],model.kernel[1]),
                            kernelderivativematrix(model.X[model.MBIndices,:],model.inducingPoints[1],model.kernel[1]),
                            kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[1])]
        (f_l,f_v) = hyperparameter_gradient_function(model)
        grads_lengthscales = compute_hyperparameter_gradient(model.kernel[1],f_l,matrix_derivatives,1,1)
        println(grads_lengthscales)
        grad_variance = f_v(model.kernel[1])
        apply_gradients_lengthscale!(model.kernel[1],grads_lengthscales)
        apply_gradients_variance!(model.kernel[1],grad_variance)
    end
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model)
        model.inducingPoints += GradDescent.update(model.optimizer,inducingpoints_gradients)
    end
    model.HyperParametersUpdated = true
end


# function setvariance(model::SparseMultiClass)
#     for k in model.KIndices
#         newvar =  model.m/(sum(model.invKmm[k].*(model.μ[k]*transpose(model.μ[k])+model.Σ[k]))-dot(model.Y[k][model.MBIndices].*model.θ[1]+model.θ[k+1],1-model.))
#         println("Variance $k : $(newvar)")
#     end
# end
#
