# Functions related to the Bayesian SVM Model cf
# "Bayesian Nonlinear Support Vector Machines for Big Data"
# Wenzel, Galy-Fajou, Deutsch and Kloft ECML 2017

function variablesUpdate_BSVM!(model::LinearBSVM,iter)
#Compute the updates for the linear BSVM
    Z = Diagonal(model.y[model.MBIndices])*model.X[model.MBIndices,:];
    model.α[model.MBIndices] = (1 - Z*model.μ).^2 +  squeeze(sum((Z*model.ζ).*Z,2),2);
    (grad_η_1,grad_η_2) = naturalGradientELBO_BSVM(model.α[model.MBIndices],Z, model.invΣ, model.Stochastic ? model.StochCoeff : 1)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function variablesUpdate_BSVM!(model::BatchBSVM,iter)
    Z = Diagonal(model.y);
    model.α = (1 - Z*model.μ).^2 +  squeeze(sum((Z*model.ζ).*Z,2),2);
    (grad_η_1,grad_η_2) = naturalGradientELBO_BSVM(model.α,Z, model.invK, 1.0)
    model.η_1 = grad_η_1; model.η_2 = grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function variablesUpdate_BSVM!(model::SparseBSVM,iter)
    Z = Diagonal(model.y[model.MBIndices])*model.κ;
    model.α[model.MBIndices] = (1 - Z*model.μ).^2 +  squeeze(sum((Z*model.ζ).*Z,2),2)+model.Ktilde;
    (grad_η_1,grad_η_2) = naturalGradientELBO_BSVM(model.α[model.MBIndices],Z, model.invKmm, model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end



function ELBO(model::LinearBSVM)
    Z = Diagonal(model.y[model.MBIndices])*model.X[model.MBIndices,:]
    ELBO = 0.5*(logdet(model.ζ)+logdet(model.invΣ)-trace(model.invΣ*(model.ζ+model.μ*transpose(model.μ))));
    for i in 1:length(model.MBIndices)
        ELBO += model.StochCoeff*(2.0*log.(model.α[model.MBIndices[i]]) + log.(besselk.(0.5,model.α[MBIndices[i]]))
        + dot(vec(Z[i,:]),model.μ) + 0.5/model.α[MBIndices[i]]*(model.α[MBIndices[i]]^2-(1-dot(vec(Z[i,:]),model.μ))^2 - dot(vec(Z[i,:]),model.ζ*vec(Z[i,:]))))
    end
    return ELBO
end

function ELBO(model::BatchBSVM)
    ELBO = 0.5*(logdet(model.ζ)+logdet(model.invK)-trace(model.invK*model.ζ)-dot(model.μ,model.invK*model.μ))
    for i in 1:model.nSamples
      ELBO += 0.25*log.(model.α[i])+log.(besselk.(0.5,sqrt.(model.α[i])))+model.y[i]*model.μ[i]+(model.α[i]-(1-model.y[i]*model.μ[i])^2-model.ζ[i,i])/(2*sqrt.(model.α[i]))
    end
    return ELBO
end

function ELBO(model::SparseBSVM)
    ELBO = 0.5*(logdet(model.ζ)+logdet(model.invKmm))
    ELBO += -0.5*(trace(model.invKmm*(model.ζ+model.μ*transpose(model.μ)))) #trace replaced by sum
    ELBO += model.StochCoeff*dot(model.y[model.MBIndices],model.κ*model.μ)
    ELBO += model.StochCoeff*sum(0.25*log.(model.α[model.MBIndices]) + log.(besselk.(0.5,sqrt.(model.α[model.MBIndices]))))
    ζtilde = model.κ*model.ζ*transpose(model.κ)
    for i in 1:length(model.MBIndices)
      ELBO += 0.5*model.StochCoeff/sqrt.(model.α[model.MBIndices[i]])*(model.α[model.MBIndices[i]]-(1-model.y[model.MBIndices[i]]*dot(model.κ[i,:],model.μ))^2-(ζtilde[i,i]+model.Ktilde[i]))
    end
    return ELBO
end

function naturalGradientELBO_BSVM(α,Z,invPrior,stoch_coef)
  grad_1 =  stoch_coef*transpose(Z)*(1./sqrt.(α)+1)
  grad_2 = -0.5*(stoch_coef*transpose(Z)*Diagonal(1./sqrt.(α))*Z + invPrior)
  (grad_1,grad_2)
end


function hyperparameter_gradient_function(model::SparseBSVM)
    B = model.μ*transpose(model.μ) + model.ζ
    Kmn = kernelmatrix(model.inducingPoints,model.X,model.kernel)
    A = Diagonal(1.0./sqrt.(model.α[model.MBIndices]))
    return function(Js)
                Jmm = Js[1]; Jnm = Js[2]; Jnn = Js[3]
                ι = (Jnm-model.κ*Jmm)*model.invKmm
                Jtilde = Jnn - sum(ι.*(Kmn.'),2) - sum(model.κ.*Jnm,2)
                V = model.invKmm*Jmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*A*model.κ + model.κ'*A*ι)) .* transpose(B)) - trace(V) - model.StochCoeff*dot(diag(A),Jtilde)
                    + 2.0*model.StochCoeff*dot(model.y[model.MBIndices],(1+A)*ι*model.μ))
            end
end

function inducingpoints_gradient(model::SparseBSVM)
    gradients_inducing_points = zeros(model.inducingPoints)
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:dim #iterate over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
            Jtilde = -sum(ι.*(Kmn.'),2)-sum(model.κ.*Jnm[j,:,:],2)
            V = model.invKmm*Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm-model.StochCoeff*(ι'*Θ*model.κ+model.κ'*Θ*ι)).*transpose(B))-trace(V)-model.StochCoeff*dot(diag(Θ),Jtilde)
             + 2.0*model.StochCoeff*dot(model.y[model.MBIndices],(1+A)*ι*model.μ))
        end
    end
    return gradients_inducing_points
end
