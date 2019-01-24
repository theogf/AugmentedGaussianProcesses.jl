"""Basic displaying function"""
function Base.show(io::IO,model::GPModel{T}) where T
    print(io,"$(model.Name){$T} model")
end

def_atfrequency = 2
def_smoothwindow = 5

"""Compute the variational updates for the full batch models"""
function variational_updates!(model::FullBatchModel{T},iter::Integer) where T
    local_update!(model)
    natural_gradient(model)
    global_update!(model)
end

"""Compute the variational updates and the new learning rate for the sparse models"""
function variational_updates!(model::SparseModel{T},iter::Integer) where T
    local_update!(model)
    (grad_η₁,grad_η₂) = natural_gradient(model)
    computeLearningRate_Stochastic!(model,iter,grad_η₁,grad_η₂);
    global_update!(model,grad_η₁,grad_η₂)
end

"""Update the global variational parameters of the linear models"""
function global_update!(model::LinearModel{T},grad_1::AbstractVector{T},grad_2::AbstractMatrix{T}) where T
    model.η₁ = (1.0-model.ρ_s)*model.η₁ + model.ρ_s*grad_1;
    model.η₂ = (1.0-model.ρ_s)*model.η₂ + model.ρ_s*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.μ = -0.5*model.η₂\model.η₁ #Back to the normal distribution parameters (needed for α updates)
    model.Σ = -0.5*inv(model.η₂);
end

"""Update the global variational parameters of the linear models"""
function global_update!(model::FullBatchModel{T}) where T
    model.Σ = inv(model.η₂)*(-0.5);
    model.μ = model.Σ*model.η₁ #Back to the normal distribution parameters (needed for α updates)
end

""""Update the global variational parameters of the sparse GP models"""
function global_update!(model::SparseModel{T},grad_1::Vector{T},grad_2::Symmetric{T,Matrix{T}}) where T
    model.η₁ = (1.0-model.ρ_s)*model.η₁ + model.ρ_s*grad_1;
    model.η₂ = Symmetric((1.0-model.ρ_s)*model.η₂ + model.ρ_s*grad_2) #Update of the natural parameters with noisy/full natural gradient
    model.Σ = -inv(model.η₂)*0.5;
    model.μ = model.Σ*model.η₁ #Back to the normal distribution parameters (needed for α updates)
end

"""Compute the KL Divergence between the GP Prior and the variational distribution for the full batch model"""
function GaussianKL(model::FullBatchModel{T}) where T
    return 0.5*(sum(model.invK.*(model.Σ+model.μ*transpose(model.μ)))-model.nSamples-logdet(model.Σ)-logdet(model.invK))
end

"""Compute the KL Divergence between the GP Prior and the variational distribution for the sparse model"""
function GaussianKL(model::SparseModel{T}) where T
    return 0.5*(sum(model.invKmm.*(model.Σ+model.μ*transpose(model.μ)))-model.m-logdet(model.Σ)-logdet(model.invKmm))
end

"""Return a function computing the gradient of the ELBO given the kernel hyperparameters for full batch Models"""
function hyperparameter_gradient_function(model::FullBatchModel{T}) where T
    A = model.invK*(model.Σ+model.µ*transpose(model.μ))-Diagonal{Float64}(I,model.nSamples)
    #return gradient functions for lengthscale, variance
    return (function(J)
                V = model.invK*J
                return 0.5*sum(V.*transpose(A))
            end,
            function(kernel)
                return 0.5/getvariance(kernel)*tr(A)
            end
            )
end
