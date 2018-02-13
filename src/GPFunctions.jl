#### Computations of the kernel matrices for the different type of models ####

function computeMatrices!(model::SparseModel)
    if model.HyperParametersUpdated
        model.invKmm = Matrix(Symmetric(inv(CreateKernelMatrix(model.inducingPoints,model.Kernel_function)+model.γ*eye(model.nFeatures))))
    end
    #If change of hyperparameters or if stochatic
    if model.HyperParametersUpdated || model.Stochastic
        Knm = CreateKernelMatrix(model.X[model.MBIndices,:],model.Kernel_function,X2=model.inducingPoints)
        model.κ = Knm*model.invKmm
        model.Ktilde = CreateDiagonalKernelMatrix(model.X[model.MBIndices,:],model.Kernel_function) + model.γ*ones(length(model.MBIndices)) - squeeze(sum(model.κ.*Knm,2),2)
    end
    model.HyperParametersUpdated=false
end

function computeMatrices!(model::FullBatchModel)
    if model.HyperParametersUpdated
        model.invK = inv(Symmetric(CreateKernelMatrix(model.X,model.Kernel_function) + model.γ*eye(model.nFeatures),:U))
        model.HyperParametersUpdated = false
    end
end

function computeMatrices!(model::LinearModel)
    if model.HyperParametersUpdated
        model.invΣ =  (1.0/model.γ)*eye(model.nFeatures)
        model.HyperParametersUpdated = false
    end
end


#### Computation of predictions with and without variance using the probit and logit link ####

function probitpredict(model::LinearModel,X_test)
    return model.Intercept ? [ones(Float64,size(X_test,1)) X_test]*model.μ : X_test*model.μ
end

function probitpredict(model::FullBatchModel,X_test)
    n = size(X_test,1)
    if model.TopMatrixForPrediction == 0
      model.TopMatrixForPrediction = model.invK*model.μ
    end
    k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    return k_star*model.TopMatrixForPrediction
end

function probitpredict(model::SparseModel,X_test)
    n = size(X_test,1)
    if model.TopMatrixForPrediction == 0
      model.TopMatrixForPrediction = model.invKmm*model.μ
    end
    k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.inducingPoints)
    return k_star*model.TopMatrixForPrediction
end

function probitpredictproba(model::LinearModel,X_test)
    if model.Intercept
      X_test = [ones(Float64,size(X_test,1)) X_test]
    end
    n = size(X_test,1)
    predic = zeros(n)
    for i in 1:n
      predic[i] = cdf(Normal(),(dot(X_test[i,:],model.μ))/(dot(X_test[i,:],model.ζ*X_test[i,:])+1))
    end
    return predic
end

function probitpredictproba(model::FullBatchModel,X_test)
    n = size(X_test,1)
    ksize = model.nSamples
    if model.DownMatrixForPrediction == 0
        if model.TopMatrixForPrediction == 0
            model.TopMatrixForPrediction = model.invK*model.μ
        end
      model.DownMatrixForPrediction = -(model.invK*(eye(ksize)-model.ζ*model.invK))
    end
    predic = zeros(n)
    k_star = zeros(ksize)
    k_starstar = 0
    for i in 1:n
      for j in 1:ksize
        k_star[j] = model.Kernel_function(model.X[j,:],X_test[i,:])
      end
      k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])
      predic[i] = cdf(Normal(),(dot(k_star,model.TopMatrixForPrediction))/(k_starstar + dot(k_star,model.DownMatrixForPrediction*k_star) + 1))
    end
    return predic
end

function probitpredictproba(model::SparseModel,X_test)
    n = size(X_test,1)
    ksize = model.m
    if model.DownMatrixForPrediction == 0
      if model.TopMatrixForPrediction == 0
        model.TopMatrixForPrediction = model.invKmm*model.μ
      end
      model.DownMatrixForPrediction = -(model.invKmm*(eye(ksize)-model.ζ*model.invKmm))
    end
    predic = zeros(n)
    k_star = zeros(ksize)
    k_starstar = 0
    for i in 1:n
      for j in 1:ksize
        k_star[j] = model.Kernel_function(model.inducingPoints[j,:],X_test[i,:])
      end
      k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])
      predic[i] = cdf(Normal(),(dot(k_star,model.TopMatrixForPrediction))/(k_starstar + dot(k_star,model.DownMatrixForPrediction*k_star) + 1))
    end
    return predic
end

function logit(x)
    return 1./(1+exp.(-x))
end

function logitpredict(model::AugmentedModel,X_test)
    y_predic = logitPredictProba(model,X_test)
    y_predic[y_predic.>0.5] = 1; y_predic[y_predic.<=0.5] = -1
    return y_predic
end

function logitpredictproba(model::FullBatchModel,X_test)
    nPoints = size(X_test,1)
    K_starN = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.X)
    K_starstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    m = K_starN*model.invK*model.μ;
    cov = K_starstar+sum((K_starN*model.invK).*transpose((model.ζ*model.invK-eye(model.nFeatures))*transpose(K_starN)),2)
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(m[i],cov[i])
        f=function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,m[i]-10*cov[i],m[i]+10*cov[i])[1]
    end
    return predic
end

function logitpredictproba(model::SparseModel,X_test)
    nPoints = size(X_test,1)
    ksize = model.m
    K_starM = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.inducingPoints)
    K_starstar = CreateDiagonalKernelMatrix(X_test,model.Kernel_function)
    m = K_starM*model.invKmm*model.μ;
    cov = K_starstar+sum((K_starM*model.invKmm).*transpose((model.ζ*model.invKmm-eye(model.nFeatures))*transpose(K_starM)),2)
    if count(cov.<=0)>0
        error("Covariance under 0, params are $(broadcast(getfield,model.Kernels,:param)) and coeffs $(broadcast(getfield,model.Kernels,:coeff))")
    end
    predic = zeros(nPoints)
    for i in 1:nPoints
        d= Normal(m[i],cov[i])
        f= function(x)
            return logit(x)*pdf(d,x)
        end
        predic[i] = quadgk(f,m[i]-10*cov[i],m[i]+10*cov[i])[1]
    end
    return predic
end

#### Get Functions ####

function getInversePrior(model::LinearModel)
    return model.invΣ
end

function getInversePrior(model::FullBatchModel)
    return model.invK
end

function getInversePrior(model::SparseModel)
    return model.invKmm
end


#### Optimization of the hyperparameters #### #TODO
function updateHyperParameters!(model::LinearModel,iter::Integer)
    grad_γ = 0.5*((trace(model.ζ)+norm(model.μ))/(model.γ^2.0)-model.nFeatures/model.γ);
    if model.VerboseLevel > 2
        println("Grad γ : $(grad_γ)")
    end
    model.γ += GradDescent.update(model.optimizers[1],grad_γ)
    model.HyperParametersUpdated = true
end

function updateHyperParameters!(model::NonLinearModel,iter::Integer)
    gradients = computeHyperParametersGradients(model,iter)
    if model.VerboseLevel > 1
        print("Hyperparameters  (param,coeff) $((getfield.(model.Kernels,:param),getfield.(model.Kernels,:coeff))) with gradients $(gradients[1:2]) \n")
    end
    applyHyperParametersGradients!(model,gradients)
    model.HyperParametersUpdated = true;
end


# Apply the gradients of the hyperparameters following Nesterov Accelerated Gradient Method and clipping method
function applyHyperParametersGradients!(model::AugmentedModel,gradients)
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
         model.inducingPoints += GradDescent.update(model.optimizers[2*model.nKernels+1],gradients[3])
    end
end

#Compute a the derivative of the covariance matrix
function computeJ(model::FullBatchModel,derivative::Function)
    return CreateKernelMatrix(model.X,derivative)
end

function computeJ(model::SparseModel,derivative::Function)
    Jnm = CreateKernelMatrix(model.X[model.MBIndices,:],derivative,X2=model.inducingPoints)
    Jnn = CreateDiagonalKernelMatrix(model.X[model.MBIndices,:],derivative)
    Jmm = CreateKernelMatrix(model.inducingPoints,derivative)
    return Jnm,Jnn,Jmm
end

function CreateColumnRowMatrix(n,iter,gradient)
    K = zeros(n,n)
    K[iter,:] = gradient; K[:,iter] = gradient;
    return K
end

function CreateColumnMatrix(n,m,iter,gradient)
    K = zeros(n,m)
    K[:,iter] = gradient;
    return K
end

#Compute the gradients given the inducing point locations
function computeIndPointsJ(model::SparseModel,iter)
    dim = size(model.X,2)
    Dnm = zeros(model.nSamplesUsed,dim)
    Dmm = zeros(model.m,dim)
    Jnm = zeros(dim,model.nSamplesUsed,model.m)
    Jmm = zeros(dim,model.m,model.m)
    function derivative(X1,X2)
        tot = 0
        for i in 1:model.nKernels
            tot += model.Kernels[i].coeff*model.Kernels[i].compute_point_deriv(X1,X2)
        end
        return tot
    end
    #Compute the gradients given every other point
    for i in 1:model.nSamplesUsed
        Dnm[i,:] = derivative(model.X[model.MBIndices[i],:],model.inducingPoints[iter,:])
    end
    for i in 1:model.m
        Dmm[i,:] = derivative(model.inducingPoints[iter,:],model.inducingPoints[i,:])
    end
    for i in 1:dim
        Jnm[i,:,:] = CreateColumnMatrix(model.nSamplesUsed,model.m,iter,Dnm[:,i])
        Jmm[i,:,:] = CreateColumnRowMatrix(model.m,iter,Dmm[:,i])
    end
    return Jnm,Jmm
end

function computeHyperParametersGradients(model::FullBatchModel,iter::Integer)
    A = model.invK*(model.ζ+model.µ*transpose(model.μ))-eye(model.nSamples)
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
