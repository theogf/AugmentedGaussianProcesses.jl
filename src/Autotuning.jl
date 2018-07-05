

#### Optimization of the hyperparameters #### #TODO
function updateHyperParameters!(model::LinearModel,iter::Integer)
    grad_noise = 0.5*((trace(model.ζ)+norm(model.μ))/(model.noise^2.0)-model.nFeatures/model.noise);
    if model.VerboseLevel > 2
        println("Grad noise : $(grad_noise)")
    end
    model.noise += GradDescent.update(model.optimizers[1],grad_noise)
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
    model.HyperParametersUpdated = true
end
function updateHyperParameters!(model::SparseModel)
    Jmm = derivativekernelmatrix(model.kernel,model.inducingPoints)
    Jnm = derivativekernelmatrix(model.kernel,model.X[model.MBIndices,:],model.inducingPoints)
    Jnn = derivativediagkernelmatrix(model.kernel,model.X[model.MBIndices,:])
    return moving_apple(model,Jmm,Jnm,Jnn)
    apply_gradients!(model.kernel,compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn]))
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model)
        model.inducingPoints -= GradDescent.update(model.optimizer,inducingpoints_gradients)
    end
    model.HyperParametersUpdated = true
end

function hyperparameter_gradient_function(model::FullBatchModel)
    A = model.invK*(model.ζ+model.µ*transpose(model.μ))-eye(model.nSamples)
    return function(Js)
                V = model.invK*Js[1]
                return 0.5*sum(V.*transpose(A))
            end
end

function apple(model::MultiClass)
    model.HyperParametersUpdated = true
    computeMatrices!(model)
    return -ELBO(model)
    # return 0.5*sum([-logdet(model.Knn)-trace(model.invK*(model.ζ[i]+model.μ[i]*transpose(model.μ[i]))) for i in 1:model.K])
end

function apple(model::GPModel)
    model.HyperParametersUpdated = true
    computeMatrices!(model)
    return -ELBO(model)
end

function acc(model,X_test,y_test)
    y_sparse, = model.predict(X_test)
    sparse_score=0
    for (i,pred) in enumerate(y_sparse)
        if pred == y_test[i]
            sparse_score += 1
        end
    end
    return sparse_score/length(y_sparse)
end
function moving_apple(model,Jmm,Jnm,Jnn)
    orig_param = copy(getvalue(model.kernel.param[1]))
    first_value=apple(model)
    hyper = (-0.1:0.01:0.1)+1; est= zeros(hyper); accu=zeros(hyper)
     for i in 1:length(hyper)
         setvalue!(model.kernel.param[1],orig_param*hyper[i])
         est[i]=apple(model)
         accu[i]=acc(model,X_test,y_test)
     end
    plot(hyper*orig_param,est)
    plot!(hyper*orig_param,accu)
    plot!([orig_param],[first_value],t=:scatter,color=:red)
    setvalue!(model.kernel.param[1],orig_param)
    apple(model)
    grads = compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn])
    apply_gradients!(model.kernel,grads)#compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jnn]))
    display(plot!([getvalue(model.kernel.param)],[apple(model)],t=:scatter,color=:green))
end
function updateHyperParameters!(model::MultiClass)
    Jnn = derivativekernelmatrix(model.kernel,model.X)
    first_value = apple(model)
    orig_param = copy(getvalue(model.kernel.param[1]))
    var_param = 1.001*orig_param
    diff = var_param-orig_param
    setvalue!(model.kernel.param[1],var_param)
    second_value = apple(model)
    println("FEM : $second_value -> $first_value, $diff")
    FEM = (second_value-first_value)/diff
    hyper = (-0.1:0.01:0.1)+1; est= zeros(hyper);
     for i in 1:length(hyper)
         setvalue!(model.kernel.param[1],orig_param*hyper[i])
         est[i]=apple(model)
         # accu[i]=acc(model,X_test,y_test)
     end
    plot(hyper*orig_param,est)
    plot!([orig_param],[first_value],t=:scatter,color=:red)
    setvalue!(model.kernel.param[1],orig_param)
    apple(model)
    grads = compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jnn])
    println("Gradients : $grads vs $FEM")
    println("Kernel before : $(getvalue(model.kernel.param))")
    println("Before hyper : $(apple(model))")
    apply_gradients!(model.kernel,grads)#compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jnn]))
    println("After hyper : $(apple(model))")
    println("Kernel after : $(getvalue(model.kernel.param))")
    display(plot!([getvalue(model.kernel.param)],[apple(model)],t=:scatter,color=:green))
end


function updateHyperParameters!(model::SparseMultiClass)
    if model.KInducingPoints
        Jmm = [derivativekernelmatrix(model.kernel,model.inducingPoints[i]) for i in 1:model.K]
        Jnm = [derivativekernelmatrix(model.kernel,model.X[model.MBIndices,:],model.inducingPoints[i]) for i in 1:model.K]
    else
        Jmm = derivativekernelmatrix(model.kernel,model.inducingPoints[1])
        Jnm = derivativekernelmatrix(model.kernel,model.X[model.MBIndices,:],model.inducingPoints[1])
    end
    Jnn = derivativediagkernelmatrix(model.kernel,model.X[model.MBIndices,:])
    first_value = apple(model)
    orig_param = copy(getvalue(model.kernel.param[1]))
    var_param = 1.001*orig_param
    diff = var_param-orig_param
    setvalue!(model.kernel.param[1],var_param)
    second_value = apple(model)
    println("FEM : $second_value -> $first_value, $diff")
    FEM = (second_value-first_value)/diff
    hyper = (-0.1:0.01:0.1)+1; est= zeros(hyper)
     for i in 1:length(hyper)
         setvalue!(model.kernel.param[1],orig_param*hyper[i])
         est[i]=apple(model)
     end
    plot(hyper*orig_param,est)
    plot!([orig_param],[first_value],t=:scatter,color=:red)
    setvalue!(model.kernel.param[1],orig_param)
    apple(model)
    grads = compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn])
    println("Gradients : $grads vs $FEM")
    println("Kernel before : $(getvalue(model.kernel.param))")
    println("Before hyper : $(apple(model))")
    apply_gradients!(model.kernel,grads)#compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jmm,Jnm,Jnn]))
    println("After hyper : $(apple(model))")
    println("Kernel after : $(getvalue(model.kernel.param))")
    display(plot!([getvalue(model.kernel.param)],[apple(model)],t=:scatter,color=:green))
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model)
        model.inducingPoints += GradDescent.update(model.optimizer,inducingpoints_gradients)
    end
    model.HyperParametersUpdated = true
end

#Printing Functions

function printautotuninginformations(model::LinearModel)
#Print the updated values of the noise
    println("Gamma : $(model.noise)")
end

function printautotuninginformations(model::NonLinearModel)
#Print the updated values of the kernel hyperparameters
    for i in 1:model.nKernels
        print("Hyperparameters  (param,coeff) $((getfield.(model.kernel,:param),getfield.(model.kernel,:coeff))) with gradients $(gradients[1:2]) \n");
    end
end
