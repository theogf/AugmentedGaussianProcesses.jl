# Functions related to the Efficient Gaussian Process Classifier (XGPC)
# https://arxiv.org/abs/1802.06383
"Update the local variational parameters of the full batch GP XGPC"
function local_update!(model::BatchXGPC)
    model.c = sqrt.(diag(model.Σ)+model.μ.^2)
    model.θ = 0.5*tanh.(0.5*model.c)./model.c
end

"Compute the variational updates for the full GP XGPC"
function variational_updates!(model::BatchXGPC,iter)
    local_update!(model)
    natural_gradient(model)
    global_update!(model)
end

"Update the local variational parameters of the sparse GP XGPC"
function local_update!(model::SparseXGPC)
    model.c = sqrt.(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2)
    model.θ = 0.5*tanh.(0.5*model.c)./model.c

end

"Compute the variational updates for the sparse GP XGPC"
function variational_updates!(model::SparseXGPC,iter::Integer)
    local_update!(model)
    (grad_η_1,grad_η_2) = natural_gradient(model)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Update the local variational parameters of the online GP XGPC"
function local_update!(model::OnlineXGPC)
    model.c = sqrt.(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2)
    model.θ = 0.5*tanh.(0.5*model.c)./model.c
end

"Compute the variational updates for the online GP XGPC"
function variational_updates!(model::OnlineXGPC,iter::Integer)
    local_update!(model)
    (grad_η_1,grad_η_2) = natural_gradient_XGPC(model)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Return the natural gradients of the ELBO given the natural parameters"
function natural_gradient(model::BatchXGPC)
    model.η_1 =  0.5*model.y
    model.η_2 = Symmetric(-0.5*(Diagonal{Float64}(model.θ) + model.invK))
end

"Return the natural gradients of the ELBO given the natural parameters"
function natural_gradient(model::SparseXGPC)
    grad_1 =  0.5*model.StochCoeff*model.κ'*model.y[model.MBIndices]
    grad_2 = Symmetric(-0.5*(model.StochCoeff*transpose(model.κ)*Diagonal{Float64}(model.θ)*model.κ .+ model.invKmm))
    return (grad_1,grad_2)
end


"Compute the negative ELBO for the full batch XGPC Model"
function ELBO(model::BatchXGPC)
    ELBO_v = ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    ELBO_v -= PolyaGammaKL(model)
    return -ELBO_v
end

"Compute the negative ELBO for the sparse XGPC Model"
function ELBO(model::SparseXGPC)
    ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    ELBO_v -= model.StochCoeff*PolyaGammaKL(model)
    return -ELBO_v
end


"Return the expected log likelihood for the batch XGPC Model"
function ExpecLogLikelihood(model::BatchXGPC)
    tot = -0.5*model.nSamples*log(2)
    tot += 0.5.*(sum(model.μ.*model.y)-sum(model.θ.*(diag(model.Σ)+model.μ.^2)))
    return tot
end

"Return the expected log likelihood for the sparse XGPC Model"
function ExpecLogLikelihood(model::SparseXGPC)
    tot = -0.5*model.nSamplesUsed*log(2)
    tot += 0.5.*(sum((model.κ*model.μ).*model.y[model.MBIndices])-sum(model.θ.*(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2)))
    return tot
end

"Return the KL divergence for the polya gamma distributions (with b_i set to 1)"
function PolyaGammaKL(model::GPModel)
    return sum(-0.5*model.c.^2 .* model.θ .+ log.(cosh.(0.5.*model.c)))
end

"Compute the negative ELBO for the sparse XGPC Model"
function ELBO(model::OnlineXGPC)
    ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    ELBO_v -= model.StochCoeff*PolyaGammaKL(model)
    return -ELBO_v
end
"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a XGPC Model"
function hyperparameter_gradient_function(model::SparseXGPC)
    F2 = Symmetric(model.μ*transpose(model.μ) + model.Σ)
    θ = Diagonal(model.θ)
    return (function(Jmm,Jnm,Jnn)
                ι = (Jnm-model.κ*Jmm)*model.invKmm
                Jtilde = Jnn - sum(ι.*model.Knm,dims=2)[:] - sum(model.κ.*Jnm,dims=2)[:]
                V = model.invKmm*Jmm
                return 0.5*(sum( (V*model.invKmm).*F2) - model.StochCoeff*sum((ι'*θ*model.κ + model.κ'*θ*ι).*F2) - tr(V) - model.StochCoeff*dot(model.θ,Jtilde)
                    + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
            end,
            function(kernel)
                return  0.5/(getvariance(kernel))*(sum(model.invKmm.*F2)-model.StochCoeff*dot(model.θ,model.Ktilde)-model.m)
            end,
            function()
                ι = -model.κ*model.invKmm
                Jtilde = ones(Float64,model.nSamplesUsed) - sum(ι.*model.Knm,dims=2)[:]
                V = model.invKmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*θ*model.κ + model.κ'*θ*ι)) .* transpose(F2)) - tr(V) - model.StochCoeff*dot(model.θ,Jtilde)
                    + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
            end)
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters"
function inducingpoints_gradient(model::SparseXGPC)
    gradients_inducing_points = zero(model.inducingPoints)
    B = model.μ*transpose(model.μ) + model.Σ
    θ = Diagonal(model.θ)
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:model.nDim #Compute the gradient over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
            Jtilde = -sum(ι.*model.Knm,dims=2)[:]-sum(model.κ.*Jnm[j,:,:],dims=2)[:]
            V = model.invKmm*Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum((V*model.invKmm).*B)
            - model.StochCoeff*sum((ι'*θ*model.κ + model.κ'*θ*ι).*B)
            - tr(V) - model.StochCoeff*dot(model.θ,Jtilde)
            + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
        end
    end
    return gradients_inducing_points
end
