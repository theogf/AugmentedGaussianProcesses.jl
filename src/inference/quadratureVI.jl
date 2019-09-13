"""
**QuadratureVI**

Variational Inference solver by approximating gradients via numerical integration via Quadrature

```julia
QuadratureVI(ϵ::T=1e-5,nGaussHermite::Integer=20,optimizer::Optimizer=Momentum(η=0.0001))
```

**Keyword arguments**

    - `ϵ::T` : convergence criteria
    - `nGaussHermite::Int` : Number of points for the integral estimation
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl) package. Default is `Momentum(η=0.0001)`
"""
mutable struct QuadratureVI{T<:Real} <: NumericalVI{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer::LatentArray{Optimizer} #Learning rate for stochastic updates
    nPoints::Int64 #Number of points for the quadrature
    nodes::Vector{T}
    weights::Vector{T}
    clipping::T
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 #Number of samples of the data
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::Vector #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::LatentArray{Vector{T}}
    ∇η₂::LatentArray{Symmetric{T,Matrix{T}}}
    ν::LatentArray{Vector{T}} #Derivative -<dv/dx>_qn
    λ::LatentArray{Vector{T}} #Derivative  <d²V/dx²>_qm
    x::SubArray{T,2,Matrix{T}}#,Tuple{UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},false}#SubArray{T,2,Matrix{T},Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true}
    y::LatentArray{SubArray}
    function QuadratureVI{T}(ϵ::T,nPoints::Integer,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,clipping::Real,nSamplesUsed::Integer=1) where T
        gh = gausshermite(nPoints)
        return new{T}(ϵ,nIter,[optimizer],nPoints,gh[1],gh[2]./sqrtπ,clipping,Stochastic,1,nSamplesUsed)
    end
end

function QuadratureVI(;ϵ::T=1e-5,nGaussHermite::Integer=100,optimizer::Optimizer=Momentum(η=1e-5),clipping::Real=0.0) where {T<:Real}
    QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,false,clipping)
end


"""
**QuadratureSVI**

Stochastic Variational Inference solver by approximating gradients via numerical integration via Quadrature

```julia
QuadratureSVI(nMinibatch::Integer;ϵ::T=1e-5,nGaussHermite::Integer=20,optimizer::Optimizer=Adam(α=0.1))
```
    -`nMinibatch::Integer` : Number of samples per mini-batches

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl) package. Default is `Momentum(η=0.001)`
"""
function QuadratureSVI(nMinibatch::Integer;ϵ::T=1e-5,nGaussHermite::Integer=20,optimizer::Optimizer=Momentum(η=1e-5),clipping::Real=0.0) where {T<:Real}
    QuadratureVI{T}(ϵ,nGaussHermite,0,optimizer,true,clipping,nMinibatch)
end

function expecLogLikelihood(model::VGP{T,L,<:QuadratureVI}) where {T,L}
    tot = 0.0
    for k in 1:model.nLatent
        for i in 1:model.nSample
            nodes = model.inference.nodes*sqrt2*sqrt(model.Σ[k][i,i]) .+ model.μ[k][i]
            tot += dot(model.inference.weights,logpdf.(model.likelihood,model.y[k][i],nodes))
        end
    end
    return tot
end

function expecLogLikelihood(model::SVGP{T,L,<:QuadratureVI}) where {T,L}
    tot = 0.0
    for k in 1:model.nLatent
        k_correct = model.nLatent == 1 ? 1 : k
        μ = model.κ[k_correct]*model.μ[k]
        Σ = opt_diag(model.κ[k_correct]*model.Σ[k],model.κ[k_correct])
        for i in 1:model.inference.nSamplesUsed
            nodes = model.inference.nodes*sqrt2*sqrt(Σ[i]) .+ μ[i]
            tot += dot(model.inference.weights,logpdf.(model.likelihood,model.inference.y[k][i],nodes))
        end
    end
    return tot
end


function compute_grad_expectations!(model::VGP{T,L,<:QuadratureVI}) where {T,L}
    for k in 1:model.nLatent
        for i in 1:model.nSample
            model.inference.ν[k][i], model.inference.λ[k][i] = grad_quad(model.likelihood, model.y[k][i], model.μ[k][i], model.Σ[k][i,i],model.inference)
        end
    end
end

function compute_grad_expectations!(model::SVGP{T,L,<:QuadratureVI}) where {T,L}
    for k in 1:model.nLatent
        k_correct = model.nLatent == 1 ? 1 : k
        μ = model.κ[k_correct]*model.μ[k]
        Σ = opt_diag(model.κ[k_correct]*model.Σ[k],model.κ[k_correct])
        for i in 1:model.inference.nSamplesUsed
            model.inference.ν[k][i], model.inference.λ[k][i] = grad_quad(model.likelihood, model.inference.y[k][i], μ[i], Σ[i], model.inference)
        end
    end
end

#Compute the first and second derivative of the log-likelihood using the quadrature nodes
function grad_quad(likelihood::Likelihood{T},y::Real,μ::Real,σ²::Real,inference::Inference) where {T<:Real}
    nodes = inference.nodes*sqrt2*sqrt(σ²) .+ μ
    Edlogpdf = dot(inference.weights,grad_log_pdf.(likelihood,y,nodes))
    Ed²logpdf = dot(inference.weights,hessian_log_pdf.(likelihood,y,nodes))
    if inference.clipping != 0
        return (abs(Edlogpdf) > inference.clipping ? sign(Edlogpdf)*inference.clipping : -Edlogpdf::T,
                abs(Ed²logpdf) > inference.clipping ? sign(Ed²logpdf)*inference.clipping : -Ed²logpdf::T)
    else
        return -Edlogpdf::T, Ed²logpdf::T
    end
end
