# Functions related to the Student T likelihood (StudentT)
"Update the local variational parameters of the full batch GP StudentT"
function local_update!(model::BatchStudentT)
    model.β = 0.5.*(diag(model.Σ)+model.μ.^2-2.0.*model.μ.*model.y+(model.y.^2).+model.ν)
    model.θ = 0.5.*(model.ν.+1.0)./model.β
end

"Compute the variational updates for the full GP StudentT"
function variational_updates!(model::BatchStudentT,iter)
    local_update!(model)
    natural_gradient_StudentT(model)
    global_update!(model)
end

"Update the local variational parameters of the sparse GP StudentT"
function local_update!(model::SparseStudentT)
    model.β = 0.5.*(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2-2.0.*(model.κ*model.μ).*model.y[model.MBIndices]+(model.y[model.MBIndices].^2).+model.ν)
    model.θ = 0.5.*(model.ν.+1.0)./model.β

end

"Compute the variational updates for the sparse GP StudentT"
function variational_updates!(model::SparseStudentT,iter::Integer)
    local_update!(model)
    (grad_η_1,grad_η_2) = natural_gradient_StudentT(model)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

# "Update the local variational parameters of the online GP StudentT"
# function local_update!(model::OnlineStudentT)
#     model.α = sqrt.(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2)
#     model.θ = 0.5*tanh.(0.5*model.α)./model.α
# end

# "Compute the variational updates for the online GP StudentT"
# function variational_updates!(model::OnlineStudentT,iter::Integer)
#     local_update!(model)
#     θ = 0.5*tanh.(0.5*model.α)./model.α
#     (grad_η_1,grad_η_2) = natural_gradient_StudentT(model)
#     computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
#     global_update!(model,grad_η_1,grad_η_2)
# end

"Return the natural gradients of the ELBO given the natural parameters"
function natural_gradient_StudentT(model::BatchStudentT)
    model.η_1 =  model.θ.*model.y
    model.η_2 = -0.5*(Diagonal{Float64}(model.θ) + model.invK)
end

"Return the natural gradients of the ELBO given the natural parameters"
function natural_gradient_StudentT(model::SparseStudentT)
    grad_1 =  model.StochCoeff*model.κ'*(model.θ.*model.y[model.MBIndices])
    grad_2 = -0.5.*(model.StochCoeff*transpose(model.κ)*Diagonal{Float64}(model.θ)*model.κ + model.invKmm)
    return (grad_1,grad_2)
end


"Compute the negative ELBO for the full batch StudentT Model"
function ELBO(model::BatchStudentT)
    ELBO_v = ExpecLogLikelihood(model)
    ELBO_v += -GaussianKL(model)
    ELBO_v += -InverseGammaKL(model)
    return -ELBO_v
end


"Compute the negative ELBO for the sparse StudentT Model"
function ELBO(model::SparseStudentT)
    ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v += -GaussianKL(model)
    ELBO_v += -model.StochCoeff*InverseGammaKL(model)
    return -ELBO_v
end

"Return the expected log likelihood for the batch StudentT Model"
function ExpecLogLikelihood(model::BatchStudentT)
    tot = -0.5*model.nSamples*log(2*π)
    tot -= 0.5.*(log.(model.β).-model.nSamples*digamma(model.α))
    tot -= 0.5.*model.α./model.β*sum(diag(model.Σ)+model.μ.^2-2*model.μ.*model.y-model.y.^2)
    return tot
end

"Return the expected log likelihood for the sparse StudentT Model"
function ExpecLogLikelihood(model::SparseStudentT)
    tot = -0.5*model.nSamplesUsed*log(2*π)
    tot -= 0.5.*(log.(model.β).-model.nSamples*digamma(model.α))
    tot -= 0.5.*model.α./model.β*sum(model.Ktilde + model.κ'*model.Σ*model.κ+(model.κ*model.μ).^2-2*(model.κ*model.μ).*model.y[model.MBIndices]-model.y[model.MBIndices].^2)
    return tot
end

"Return the KL divergence for the inverse gamma distributions"
function InverseGammaKL(model::GPModel)
    α_p = β_p = model.ν/2;
    return (model.α-α_p)*digamma(model.α).-log(gamma(model.α)).+log(gamma(α_p))
            .+ α_p.*(log.(model.β).-log(β_p)).+model.α.*(β_p.-model.β)./model.β
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a StudentT Model"
function hyperparameter_gradient_function(model::SparseStudentT)
    F2 = Symmetric(model.μ*transpose(model.μ) + model.Σ)
    θ = Diagonal(model.θ)
    return (function(Jmm,Jnm,Jnn)
                ι = (Jnm-model.κ*Jmm)*model.invKmm
                Jtilde = Jnn - sum(ι.*model.Knm,dims=2)[:] - sum(model.κ.*Jnm,dims=2)[:]
                V = model.invKmm*Jmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*θ*model.κ + model.κ'*θ*ι)) .* transpose(F2)) - tr(V) - model.StochCoeff*dot(model.θ,Jtilde)
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
function inducingpoints_gradient(model::SparseStudentT)
    gradients_inducing_points = zeros(model.inducingPoints)
    B = model.μ*transpose(model.μ) + model.Σ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    θ = Diagonal(0.25./model.α.*tanh.(0.5*model.α))
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:model.nDim #Compute the gradient over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])/model.Kmm
            Jtilde = -sum(ι.*(transpose(Kmn)),dims=2)[:]-sum(model.κ.*Jnm[j,:,:],dims=2)[:]
            V = model.Kmm\Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum( (V/model.Kmm - model.StochCoeff*(ι'*θ*model.κ + model.κ'*θ*ι)) .* transpose(B)) - tr(V) - model.StochCoeff*dot(diag(θ),Jtilde)
                + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
        end
    end
    return gradients_inducing_points
end
