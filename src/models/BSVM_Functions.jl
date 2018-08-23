# Functions related to the Bayesian SVM Model cf
# "Bayesian Nonlinear Support Vector Machines for Big Data"
# Wenzel, Galy-Fajou, Deutsch and Kloft ECML 2017

"Update the local variational parameters of the linear BSVM"
function local_update!(model::LinearBSVM,Z::Matrix{Float64})
    model.α = (1.0 .- Z*model.μ).^2 +  dropdims(sum((-0.5*Z/model.η_2).*Z,dims=2),dims=2);
end

"Compute the variational updates for the linear BSVM"
function variational_updates!(model::LinearBSVM,iter::Integer)
    Z = Diagonal{Float64}(model.y[model.MBIndices])*model.X[model.MBIndices,:];
    local_update!(model,Z)
    (grad_η_1,grad_η_2) = natural_gradient_BSVM(model.α,Z, model.invΣ, model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Update the local variational parameters of full batch GP BSVM"
function local_update!(model::BatchBSVM,Z::Diagonal{Float64})
    model.α = (1.0 .- Z*model.μ).^2 +  dropdims(sum((-0.5*Z/model.η_2).*Z,dims=2),dims=2);
end

"Compute the variational updates for the full GP BSVM"
function variational_updates!(model::BatchBSVM,iter::Integer)
    Z = Diagonal{Float64}(model.y);
    local_update!(model,Z)
    (model.η_1,model.η_2) = natural_gradient_BSVM(model.α,Z,model.invK, 1.0)
    global_update!(model)
end

"Update the local variational parameters of the sparse GP BSVM"
function local_update!(model::SparseBSVM,Z::Matrix{Float64})
    model.α = (1 .- Z*model.μ).^2 + sum((-0.5*Z/model.η_2).*Z,dims=2)[:] + model.Ktilde;
end

"Compute the variational updates for the sparse GP BSVM"
function variational_updates!(model::SparseBSVM,iter::Integer)
    Z = Diagonal{Float64}(model.y[model.MBIndices])*model.κ;
    local_update!(model,Z)
    (grad_η_1,grad_η_2) = natural_gradient_BSVM(model.α,Z, model.invKmm, model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Return the natural gradients of the ELBO given the natural parameters"
function natural_gradient_BSVM(α::Vector{Float64},Z::AbstractArray{Float64,2},invPrior::Matrix{Float64},stoch_coef::Float64)
  grad_1 =  stoch_coef*transpose(Z)*(1.0./sqrt.(α).+1.0)
  grad_2 = -0.5*(stoch_coef*transpose(Z)*Diagonal(1.0./sqrt.(α))*Z + invPrior)
  return (grad_1,grad_2)
end

"Compute the negative ELBO for the linear BSVM Model"
function ELBO(model::LinearBSVM)
    Z = Diagonal{Float64}(model.y[model.MBIndices])*model.X[model.MBIndices,:]
    ELBO = 0.5*(logdet(model.Σ)+logdet(model.invΣ)-tr(model.invΣ*(model.Σ+model.μ*transpose(model.μ))));
    ELBO += sum(model.StochCoeff*(2.0*log.(model.α) + log.(besselk.(0.5,model.α))
        + dot(vec(Z[i,:]),model.μ) + 0.5./model.α.*(model.α.^2-(1-dot(vec(Z[i,:]),model.μ))^2 - dot(vec(Z[i,:]),model.Σ*vec(Z[i,:])))))
    return -ELBO
end

"Compute the ELBO for the full batch GP BSVM Model"
function ELBO(model::BatchBSVM) #TODO THERE IS A PROBLEM WITH THE ELBO COMPUTATION
    ELBO = 0.5*(logdet(model.Σ)+logdet(model.invK)-sum(model.invK.*transpose(model.Σ+model.μ*transpose(model.μ))))
    ELBO += sum(0.25*log.(model.α[i])+log.(besselk.(0.5,sqrt.(model.α)))+model.y.*model.μ+(model.α-(1-model.y.*model.μ[i]).^2-diag(model.Σ))./(2*sqrt.(model.α)))
    return -ELBO
end

"Compute the ELBO for the sparse GP BSVM Model"
function ELBO(model::SparseBSVM)#TODO THERE IS A PROBLEM WITH THE ELBO COMPUTATION
    ELBO = 0.5*(logdet(model.Σ)+logdet(model.invKmm))
    ELBO += -0.5*(tr(model.invKmm*(model.Σ+model.μ*transpose(model.μ)))) #trace replaced by sum
    ELBO += model.StochCoeff*dot(model.y[model.MBIndices],model.κ*model.μ)
    ELBO += model.StochCoeff*sum(0.25*log.(model.α[model.MBIndices]) + log.(besselk.(0.5,sqrt.(model.α[model.MBIndices]))))
    Σtilde = model.κ*model.Σ*transpose(model.κ)
    ELBO += 0.5*model.StochCoeff/sqrt.(model.α).*(model.α[model.MBIndices[i]]-(1-model.y.*dot(model.κ[i,:],model.μ)).^2-(diag(Σtilde)+model.Ktilde))
    return -ELBO
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a BSVM Model"
function hyperparameter_gradient_function(model::SparseBSVM)
    B = model.μ*transpose(model.μ) + model.Σ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    A = Diagonal(1.0./sqrt.(model.α))
    return function(Js,iter)
                Jmm = Js[1]; Jnm = Js[2]; Jnn = Js[3]
                ι = (Jnm-model.κ*Jmm)*model.invKmm
                Jtilde = Jnn - sum(ι.*transpose(Kmn),dims=2) - sum(model.κ.*Jnm,dims=2)
                V = model.invKmm*Jmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*A*model.κ + model.κ'*A*ι)) .* transpose(B)) - tr(V) - model.StochCoeff*dot(diag(A),Jtilde)
                    + 2.0*model.StochCoeff*dot(model.y[model.MBIndices],(Diagonal{Float64}(I,model.nSamplesUsed)+A)*ι*model.μ))
            end
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters"
function inducingpoints_gradient(model::SparseBSVM)
    gradients_inducing_points = zero(model.inducingPoints)
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:dim #iterate over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
            Jtilde = -sum(ι.*transpose(Kmn),dims=2)-sum(model.κ.*Jnm[j,:,:],dims=2)
            V = model.invKmm*Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm-model.StochCoeff*(ι'*Θ*model.κ+model.κ'*Θ*ι)).*transpose(B))-tr(V)-model.StochCoeff*dot(diag(Θ),Jtilde)
             + 2.0*model.StochCoeff*dot(model.y[model.MBIndices],(1+A)*ι*model.μ))
        end
    end
    return gradients_inducing_points
end
