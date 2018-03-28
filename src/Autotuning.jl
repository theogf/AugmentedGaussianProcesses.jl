

#### Optimization of the hyperparameters #### #TODO
function updateHyperParameters!(model::LinearModel,iter::Integer)
    grad_γ = 0.5*((trace(model.ζ)+norm(model.μ))/(model.γ^2.0)-model.nFeatures/model.γ);
    if model.VerboseLevel > 2
        println("Grad γ : $(grad_γ)")
    end
    model.γ += GradDescent.update(model.optimizers[1],grad_γ)
    model.HyperParametersUpdated = true
end

#NOT USED AT THE MOMENT Apply the gradients of the hyperparameters following Nesterov Accelerated Gradient Method and clipping method
function applyHyperParametersGradients!(model::GPModel,gradients)
    #Gradients contain the : kernel param gradients, kernel coeffs gradients and eventually the inducing points gradients
    for i in 1:model.nKernels
        model.Kernels[i].param += GradDescent.update(model.optimizers[i],gradients[1][i])
        model.Kernels[i].coeff += GradDescent.update(model.optimizers[i+model.nKernels],gradients[2][i])
        #Put a limit on the kernel coefficient value
        model.Kernels[i].coeff = model.Kernels[i].coeff > 0 ? model.Kernels[i].coeff : 0;
    end
    #Avoid the case where the coeff of a kernel overshoot to 0
    if model.nKernels == 1 && model.Kernels[1].coeff < 1e-14
        model.Kernels[1].coeff = 1e-12
    end
    if length(gradients)==3
         model.inducingPoints += GradDescent.update(model.optimizer,gradients[3])
    end
end

function updateHyperParameters!(model::FullBatchModel)
    Jnn = derivativekernelmatrix(model.kernel,model.X)
    apply_gradients!(model.kernel,compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jnn]))
end
function updateHyperParameters!(model::SparseModel)
    Jmm = derivativekernelmatrix(model.kernel,model.inducingPoints)
    Jnm = derivativekernelmatrix(model.kernel,model.X[model.MBIndices,:],model.inducingPoints)
    Jnn = derivativediagkernelmatrix(model.kernel,model.X[model.MBIndices,:])
    apply_gradients!(model.kernel,compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn]))
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model)
        model.inducingPoints += GradDescent.update(model.optimizer,inducingpoints_gradients)
    end
end


function hyperparameter_gradient_function(model::FullBatchModel)
    A = model.invK*(model.ζ+model.µ*transpose(model.μ))-eye(model.nSamples)
    return function(Js)
                V = model.invK*Js[1]
                return 0.5*sum(V.*transpose(A))
            end
end

#Printing Functions

function printautotuninginformations(model::LinearModel)
#Print the updated values of the noise
    println("Gamma : $(model.γ)")
end

function printautotuninginformations(model::NonLinearModel)
#Print the updated values of the kernel hyperparameters
    for i in 1:model.nKernels
        print("Hyperparameters  (param,coeff) $((getfield.(model.Kernels,:param),getfield.(model.Kernels,:coeff))) with gradients $(gradients[1:2]) \n");
    end
end
