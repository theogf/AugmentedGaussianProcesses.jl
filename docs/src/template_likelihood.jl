"""
**Template Likelihood**

Template file for likelihood creation

```julia
TemplateLikelihood()
```
See all functions you need to implement
---


"""
struct TemplateLikelihood{T<:Real,A<:AbstractVector{T}} <: AbstractLikelihood{T}
    ## Additional parameters can be added
    θ::A
    function TemplateLikelihood{T}() where {T<:Real}
        return new{T,Vector{T}}()
    end
    function TemplateLikelihood{T}(θ::A) where {T<:Real,A<:AbstractVector{T}}
        return new{T,A}(θ)
    end
end

function TemplateLikelihood()
    return TemplateLikelihood{Float64}()
end

function implemented(
    ::TemplateLikelihood, ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
)
    return true
end

function pdf(l::TemplateLikelihood, y::Real, f::Real) 
end

function Base.show(io::IO, model::TemplateLikelihood)
    return print(io, "Template Likelihood")
end

function compute_proba(
    l::TemplateLikelihood{T}, μ::AbstractVector, σ²::AbstractVector
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    return pred, sig_pred
end

### Local Updates Section ###

function local_updates!(
    l::TemplateLikelihood{T}, y::AbstractVector, μ::AbstractVector, diagΣ::AbstractVector
) 
end

function sample_local!(
    l::TemplateLikelihood{T}, y::AbstractVector, f::AbstractVector
) where {T}
    return nothing
end

### Natural Gradient Section ###

∇E_μ(l::TemplateLikelihood, ::AOptimizer, y::AbstractVector) = (nothing,)
∇E_Σ(l::TemplateLikelihood, ::AOptimizer, y::AbstractVector) = (nothing,)
