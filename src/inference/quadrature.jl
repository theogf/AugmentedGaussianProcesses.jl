mutable struct QuadratureInference{T<:Real} <: NumericalInference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer_η₁::LatentArray{Optimizer} #Learning rate for stochastic updates
    optimizer_η₂::LatentArray{Optimizer} #Learning rate for stochastic updates
    nPoints::Int64 #Number of points for the quadrature
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 #Number of samples of the data
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::Vector #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::LatentArray{AbstractVector{T}}
    ∇η₂::LatentArray{AbstractArray{T}}
    ∇μE::LatentArray{AbstractVector{T}}
    ∇ΣE::LatentArray{AbstractVector{T}}
    function QuadratureInference{T}(ϵ::T,nPoints::Integer,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamplesUsed::Integer=1) where T
        return new{T}(ϵ,nIter,[optimizer],[optimizer],nPoints,Stochastic,1,nSamplesUsed)
    end
end

function compute_grad_expectations!(model::VGP{<:Likelihood,<:QuadratureInference})
    for k in 1:model.nLatent
        for i in 1:model.nSample
            grad_quad(model.likelihood,model.y[k][i],model.μ[k][i],model.Σ[k][i,i])
        end
    end
end

function compute_grad_expectations!(model::SVGP{<:Likelihood,<:QuadratureInference})
    μ = model.κ.*model.μ; Σ = opt_diag(model.κ.*model.Σ,model.κ)
    for k in 1:model.nLatent
        for i in 1:model.nSample
            model.inference.∇μE[k][i], model.inference.∇ΣE[k][i] = grad_quad(model.likelihood,model.y[k][i],μ[k][i],Σ[k][i,i])
        end
    end
end

function grad_quad(likelihood::Likelihood,y::Real,μ::Real,σ²::Real,nPoints::Int) where {T<:Real}
    e = expectation(Normal(μ,sqrt(σ²)),n=nPoints)
    p = e(x->pdf(likelihood,y,x),d)
    dE = e(x->gradpdf(likelihood,y,x))/p
    d²E = e(x->hessiandiagpdf(likelihood,y,x))/p
    return dE, d²E - dE^2
end


function compute_log_expectations(model::VGP{<:Likelihood,<:QuadratureInference})
end


function compute_log_expectations(model::SVGP{<:Likelihood,<:QuadratureInference})
end
