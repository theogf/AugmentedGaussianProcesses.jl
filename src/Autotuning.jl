

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
    apply_gradients!(model.kernel,compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),true,Any[Jmm,Jnm,Jnn],1))
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model) #Compute the gradient given the inducing points location
        model.inducingPoints -= GradDescent.update(model.optimizer,inducingpoints_gradients) #Apply the gradients on the location
    end
    model.HyperParametersUpdated = true
end


"Update all hyperparameters for the multiclass GP models"
function updateHyperParameters!(model::MultiClass)
    if model.IndependentGPs
        Jnn = [[derivativekernelmatrix(model.kernel[i],model.X)] for i in 1:model.K]
        grads = compute_hyperparameter_gradient.(model.kernel,hyperparameter_gradient_function(model),true,Jnn,1:model.K)
    else
        Jnn = [[derivativekernelmatrix(model.kernel[1],model.X)]]
        grads = compute_hyperparameter_gradient.(model.kernel,hyperparameter_gradient_function(model),true,Jnn,1)
    end
    # broadcast(x->println(getvalue(x.param)),model.kernel)
    apply_gradients!.(model.kernel,grads)#compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn]))
    model.HyperParametersUpdated = true
end


function updateHyperParameters!(model::SparseMultiClass)
    if model.IndependentGPs
        matrix_derivatives = [[derivativekernelmatrix(model.kernel[i],model.inducingPoints[i]),
                            derivativekernelmatrix(model.kernel[i],model.X[model.MBIndices,:],model.inducingPoints[i]),
                            derivativediagkernelmatrix(model.kernel[i],model.X[model.MBIndices,:])] for i in 1:model.K]
        grads = compute_hyperparameter_gradient.(model.kernel,hyperparameter_gradient_function(model),true,matrix_derivatives,1:model.K)
    else
        matrix_derivatives = [[derivativekernelmatrix(model.kernel[1],model.inducingPoints[1]),
                            derivativekernelmatrix(model.kernel[1],model.X[model.MBIndices,:],model.inducingPoints[1]),
                            derivativediagkernelmatrix(model.kernel[1],model.X[model.MBIndices,:])]]
        grads = compute_hyperparameter_gradient.(model.kernel,hyperparameter_gradient_function(model),true,matrix_derivatives,1)
    end
    # println("Hyperparameters grads : $grads")
    apply_gradients!.(model.kernel,grads)
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model)
        model.inducingPoints += GradDescent.update(model.optimizer,inducingpoints_gradients)
    end
    model.HyperParametersUpdated = true
end
