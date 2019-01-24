### Functions related to the BSVM Likelihood ###

"""Update the local variational parameters of the linear BSVM"""
function local_update!(model::LinearBSVM{T},Z::Matrix{T}) where T
    model.α = (1.0 .- Z*model.μ).^2 +  dropdims(sum((-0.5*Z/model.η₂).*Z,dims=2),dims=2);
end

"""Compute the variational updates for the linear BSVM"""
function variational_updates!(model::LinearBSVM{T},iter::Integer) where T
    Z = Diagonal{Float64}(model.y[model.MBIndices])*model.X[model.MBIndices,:];
    local_update!(model,Z)
    (grad_η₁,grad_η₂) = natural_gradient_BSVM(model.α,Z, model.invΣ, model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η₁,grad_η₂);
    global_update!(model,grad_η₁,grad_η₂)
end

"""Update the local variational parameters of full batch GP BSVM"""
function local_update!(model::BatchBSVM{T}) where T
    model.α = (1.0 .- model.y.*model.μ).^2 .+ diag(model.Σ);
end

"""Update the local variational parameters of the sparse GP BSVM"""
function local_update!(model::SparseBSVM{T}) where T
    model.α = (1.0 .- Diagonal(model.y[model.MBIndices])*model.κ*model.μ).^2 + sum((model.κ*model.Σ).*model.κ,dims=2)[:] .+ model.Ktilde;
end

"""Return the natural gradients of the ELBO given the natural parameters"""
function natural_gradient(model::BatchBSVM{T}) where T
  model.η₁ =  model.y.*(1.0./sqrt.(model.α).+1.0)
  model.η₂ = Symmetric(-0.5*(Diagonal(1.0./sqrt.(model.α)) + model.invK))
end

"""Return the natural gradients of the ELBO given the natural parameters"""
function natural_gradient(model::SparseBSVM{T}) where T
  grad_1 =  model.StochCoeff*model.κ'*(model.y[model.MBIndices].*(1.0./sqrt.(model.α).+1.0))
  grad_2 = Symmetric(-0.5*(model.StochCoeff*model.κ'*Diagonal(1.0./sqrt.(model.α))*model.κ + model.invKmm))
  return (grad_1,grad_2)
end

"""Compute the negative ELBO for the linear BSVM Model"""
function ELBO(model::LinearBSVM{T}) where T
    Z = Diagonal{Float64}(model.y[model.MBIndices])*model.X[model.MBIndices,:]
    ELBO = 0.5*(logdet(model.Σ)+logdet(model.invΣ)-tr(model.invΣ*(model.Σ+model.μ*transpose(model.μ))));
    ELBO += sum(model.StochCoeff*(2.0*log.(model.α) + log.(besselk.(0.5,model.α))
        + dot(vec(Z[i,:]),model.μ) + 0.5./model.α.*(model.α.^2-(1-dot(vec(Z[i,:]),model.μ))^2 - dot(vec(Z[i,:]),model.Σ*vec(Z[i,:])))))
    return -ELBO
end

"""Compute the ELBO for the full batch GP BSVM Model"""
function ELBO(model::BatchBSVM{T}) where T #TODO THERE IS A PROBLEM WITH THE ELBO COMPUTATION
    # print("\n")
    ELBO_v = ExpecLogLikelihood(model)
    # println(ExpecLogLikelihood(model))
    ELBO_v -= GaussianKL(model)
    # println(GaussianKL(model))
    ELBO_v -= GIGKL(model)
    # println(GIGKL(model))
    # ELBO = 0.5*(logdet(model.Σ)+logdet(model.invK)-sum(model.invK.*transpose(model.Σ+model.μ*transpose(model.μ))))
    # ELBO += sum(0.25*log.(model.α[i])+log.(besselk.(0.5,sqrt.(model.α)))+model.y.*model.μ+(model.α-(1-model.y.*model.μ[i]).^2-diag(model.Σ))./(2*sqrt.(model.α)))
    return -ELBO_v
end

"""Compute the ELBO for the sparse GP BSVM Model"""
function ELBO(model::SparseBSVM{T}) where T#TODO THERE IS A PROBLEM WITH THE ELBO COMPUTATION
    ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    ELBO_v -= model.StochCoeff*GIGKL(model)
    # ELBO = 0.5*(logdet(model.Σ)+logdet(model.invKmm))
    # ELBO += -0.5*(tr(model.invKmm*(model.Σ+model.μ*transpose(model.μ)))) #tr replaced by sum
    # ELBO += model.StochCoeff*dot(model.y[model.MBIndices],model.κ*model.μ)
    # ELBO += model.StochCoeff*sum(0.25*log.(model.α[model.MBIndices]) + log.(besselk.(0.5,sqrt.(model.α[model.MBIndices]))))
    # Σtilde = model.κ*model.Σ*transpose(model.κ)
    # ELBO += 0.5*model.StochCoeff/sqrt.(model.α).*(model.α[model.MBIndices[i]]-(1-model.y.*dot(model.κ[i,:],model.μ)).^2-(diag(Σtilde)+model.Ktilde))
    return -ELBO_v
end

"""Return the expected log likelihood for the batch BSVM Model"""
function ExpecLogLikelihood(model::BatchBSVM{T}) where T
    tot = -model.nSamples*(0.5*log(2π)+1)
    tot += sum(model.y.*model.μ - 0.5*((1.0.-model.y.*model.μ).^2+diag(model.Σ))./sqrt.(model.α))
    # tot = sum((model.Ktilde+(1.0-model.y.*model.μ).^2+)./sqrt.(model.α))
    return tot
end

"""Return the expected log likelihood for the sparse BSVM Model"""
function ExpecLogLikelihood(model::SparseBSVM{T}) where T
    tot = -model.nSamplesUsed*(0.5*log(2π)+1)
    tot += sum(model.y[model.MBIndices].*(model.κ*model.μ) - 0.5*((1.0.-model.y[model.MBIndices].*(model.κ*model.μ)).^2+model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:])./model.α)
    # tot = sum((model.Ktilde+(1.0-model.y.*model.μ).^2+)./sqrt.(model.α))
    return tot
end

"""Return the KL divergence for the Generalized Inverse Gaussian distributions (for the improper prior p(lambda)=1)"""
function GIGKL(model::GPModel{T}) where T
    return -0.25*sum(model.α)-sum(log.(besselk.(0.5,sqrt.(model.α))))-0.5*sum(sqrt.(model.α))
end

"""Return a function computing the gradient of the ELBO given the kernel hyperparameters for a BSVM Model"""
function hyperparameter_gradient_function(model::SparseBSVM{T}) where T
    F2 = Symmetric(model.μ*transpose(model.μ) + model.Σ)
    A = Diagonal(1.0./sqrt.(model.α))
    return (function(Jmm,Jnm,Jnn)
                ι = (Jnm-model.κ*Jmm)*model.invKmm
                Jtilde = Jnn - sum(ι.*model.Knm,dims=2)[:] - sum(model.κ.*Jnm,dims=2)[:]
                V = model.invKmm*Jmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*A*model.κ + model.κ'*A*ι)) .* F2) - tr(V) - model.StochCoeff*dot(diag(A),Jtilde)
                    + 2.0*model.StochCoeff*dot(model.y[model.MBIndices],(Diagonal{Float64}(I,model.nSamplesUsed)+A)*ι*model.μ))
            end,
            function(kernel)
                0.5/(getvariance(kernel))*(sum(model.invKmm.*F2)-model.StochCoeff*dot(diag(A),model.Ktilde)-model.m)
            end,
            function()
                ι = -model.κ*model.invKmm
                Jtilde = ones(Float64,model.nSamplesUsed) - sum(ι.*model.Knm,dims=2)[:]
                V = model.invKmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*A*model.κ + model.κ'*A*ι)) .* F2) - tr(V) - model.StochCoeff*dot(diag(A),Jtilde)
                    .+ 2.0*model.StochCoeff*dot(model.y[model.MBIndices],(Diagonal{Float64}(I,model.nSamplesUsed)+A)*ι*model.μ))
            end)
end

"""Return a function computing the gradient of the ELBO given the inducing point locations"""
function inducingpoints_gradient(model::SparseBSVM{T}) where T
    gradients_inducing_points = zero(model.inducingPoints)
    B = model.μ*transpose(model.μ) + model.Σ
    A = Diagonal(1.0./sqrt.(model.α))
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:model.nDim #iterate over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
            Jtilde = -sum(ι.*model.Knm,dims=2)-sum(model.κ.*Jnm[j,:,:],dims=2)
            V = model.invKmm*Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm).*B)
            - model.StochCoeff*sum((ι'*A*model.κ+model.κ'*A*ι).*B)
            - tr(V) - model.StochCoeff*dot(diag(A),Jtilde)
             + 2.0*model.StochCoeff*dot(model.y[model.MBIndices],(I+A)*ι*model.μ))
        end
    end
    return gradients_inducing_points
end
