"""
**Template Likelihood**

Template file for likelihood creation

```julia
TemplateLikelihood()
```
See all functions you need to implement
---


"""
struct TemplateLikelihood{T<:Real} <: AbstractLikelihood{T}
    ## Additional parameters can be added
end

function TemplateLikelihood()
    return TemplateLikelihood{Float64}()
end

function implemented(
    ::TemplateLikelihood, ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
)
    return true
end

function pdf(l::TemplateLikelihood, y::Real, f::Real) end

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
    local_vars,
    l::TemplateLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    # Update the local variables here (in place if you want)
    return local_vars
end

function sample_local!(
    local_vars, l::TemplateLikelihood{T}, y::AbstractVector, f::AbstractVector
) where {T}
    # Sample the local variables here (in place if you want)
    return local_vars
end

### Natural Gradient Section ###

∇E_μ(l::TemplateLikelihood, ::AOptimizer, y, state) = (nothing,)
∇E_Σ(l::TemplateLikelihood, ::AOptimizer, y, state) = (nothing,)
