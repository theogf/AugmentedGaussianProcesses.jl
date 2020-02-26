"""
**QuadratureVI**

Variational Inference solver by approximating gradients via numerical integration via Quadrature

```julia
QuadratureVI(ϵ::T=1e-5,nGaussHermite::Integer=20,optimiser=Momentum(0.0001))
```

**Keyword arguments**

    - `ϵ::T` : convergence criteria
    - `nGaussHermite::Int` : Number of points for the integral estimation
    - `natural::Bool` : Use natural gradients
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.0001)`
"""
mutable struct QuadratureVI{T<:Real,N} <: NumericalVI{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    nPoints::Int64 #Number of points for the quadrature
    nodes::Vector{T}
    weights::Vector{T}
    clipping::T
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 #Number of samples of the data
    nMinibatch::Int64 #Size of mini-batches
    NaturalGradient::Bool
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::NTuple{N,NVIOptimizer}
    MBIndices::Vector #Indices of the minibatch
    xview::SubArray{T,2,Matrix{T}}
    yview::SubArray

    function QuadratureVI{T}(ϵ::T,nPoints::Integer,optimiser,Stochastic::Bool,clipping::Real,nMinibatch::Int,natural::Bool) where {T}
        return new{T,1}(ϵ,0,nPoints,[],[],clipping,Stochastic,0,nMinibatch,natural,1.0,true,(NVIOptimizer{T}(0,0,optimiser),))
    end

    function QuadratureVI{T,1}(ϵ::T,Stochastic::Bool,nPoints::Int,clipping::Real,nFeatures::Int,nSamples::Int,nMinibatch::Int,nLatent::Int,optimiser,natural::Bool) where {T}
        gh = gausshermite(nPoints)
        vi_opts = ntuple(_->NVIOptimizer{T}(nFeatures,nMinibatch,optimiser),nLatent)
        new{T,nLatent}(ϵ,0,nPoints,gh[1].*sqrt2,gh[2]./sqrtπ,clipping,Stochastic,nSamples,nMinibatch,natural,nSamples/nMinibatch,true,vi_opts,collect(1:nMinibatch))
    end
end

function QuadratureVI(;ϵ::T=1e-5,nGaussHermite::Integer=100,optimiser=Momentum(1e-5),clipping::Real=0.0,natural::Bool=true) where {T<:Real}
    QuadratureVI{T}(ϵ,nGaussHermite,optimiser,false,clipping,1,natural)
end


"""
**QuadratureSVI**

Stochastic Variational Inference solver by approximating gradients via numerical integration via Quadrature

```julia
QuadratureSVI(nMinibatch::Integer;ϵ::T=1e-5,nGaussHermite::Integer=20,optimiser=Momentum(0.0001))
```
    -`nMinibatch::Integer` : Number of samples per mini-batches

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nGaussHermite::Int` : Number of points for the integral estimation (for the QuadratureVI)
    - `natural::Bool` : Use natural gradients
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.0001)`
"""
function QuadratureSVI(nMinibatch::Integer;ϵ::T=1e-5,nGaussHermite::Integer=100,optimiser=Momentum(1e-5),clipping::Real=0.0) where {T<:Real}
    QuadratureVI{T}(ϵ,nGaussHermite,optimiser,clipping,nMinibatch)
end

function tuple_inference(i::TInf,nLatent::Integer,nFeatures::Integer,nSamples::Integer,nMinibatch::Integer) where {TInf <: QuadratureVI}
    return TInf(i.ϵ,i.Stochastic,i.nPoints,i.clipping,nFeatures,nSamples,nMinibatch,nLatent,i.vi_opt[1].optimiser,i.NaturalGradient)
end

function expec_log_likelihood(model::VGP{T,L,<:QuadratureVI}) where {T,L}
    tot = 0.0
    for gp in model.f
        for i in 1:model.nSamples
            x = model.inference.nodes*sqrt(max(gp.Σ[i,i],0.0)) .+ gp.μ[i]
            tot += dot(model.inference.weights,logpdf.(model.likelihood,model.y[i],x))
        end
    end
    return tot
end

function expec_log_likelihood(l::Likelihood,i::QuadratureVI,y,μ::AbstractVector,Σ::AbstractVector)
    sum(apply_quad.(y,μ,Σ,i,l))
end

function apply_quad(y::Real,μ::Real,σ²::Real,i::QuadratureVI,l::Likelihood) where {T}
    x = i.nodes*sqrt(max(σ², zero(σ²))) .+ μ
    return dot(i.weights,x)
    # return dot(i.weights,logpdf.(l,y,x))
end

function expec_log_likelihood(model::SVGP{T,L,<:QuadratureVI}) where {T,L}
    tot = 0.0
    y = get_y(model)
    for gp in model.f
        μ = mean_f(gp)
        Σ = opt_diag(gp.κ*gp.Σ,gp.κ)
        for i in 1:model.inference.nMinibatch
            x = model.inference.nodes*sqrt(max(Σ[i],zero(T))) .+ μ[i]
            tot += dot(model.inference.weights,logpdf.(model.likelihood,y[i],x))
        end
    end
    return tot
end


function compute_grad_expectations!(model::AbstractGP{T,L,<:QuadratureVI}) where {T,L}
    y = get_y(model)
    for (gp,opt) in zip(model.f,model.inference.vi_opt)
        μ = mean_f(gp)
        Σ = diag_cov_f(gp)
        for i in 1:model.inference.nMinibatch
            opt.ν[i], opt.λ[i] = grad_quad(model.likelihood,y[i],μ[i],Σ[i],model.inference)
        end
    end
end

#Compute the first and second derivative of the log-likelihood using the quadrature nodes
function grad_quad(l::Likelihood{T},y::Real,μ::Real,σ²::Real,i::Inference) where {T<:Real}
    x = i.nodes*sqrt(max(σ²,zero(T))) .+ μ
    Edlogpdf = dot(i.weights,grad_logpdf.(l,y,x))
    Ed²logpdf = dot(i.weights,hessian_logpdf.(l,y,x))
    if i.clipping != 0
        return (abs(Edlogpdf) > i.clipping ? sign(Edlogpdf)*i.clipping : -Edlogpdf::T,
                abs(Ed²logpdf) > i.clipping ? sign(Ed²logpdf)*i.clipping : -Ed²logpdf::T)
    else
        return -Edlogpdf::T, Ed²logpdf::T
    end
end
