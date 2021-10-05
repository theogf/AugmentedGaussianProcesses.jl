struct EmpiricalMean{T<:Real,V<:AbstractVector{<:Real},O} <: PriorMean{T}
    C::V
    opt::O
end

"""
    EmpiricalMean(c::AbstractVector{<:Real}=1.0;opt=ADAM(0.01))

## Arguments
- `c::AbstractVector` : Empirical mean vector

Construct a empirical mean with values `c`
Optionally give an optimiser `opt` (`ADAM(0.01)` by default)
"""
function EmpiricalMean(c::V; opt=ADAM(0.01)) where {V<:AbstractVector{<:Real}}
    return EmpiricalMean{eltype(c),V,typeof(opt)}(copy(c), opt)
end

function Base.show(io::IO, ::MIME"text/plain", μ₀::EmpiricalMean)
    return print(io, "Empirical Mean Prior (length(c) = ", length(μ₀.C), ")")
end

function (μ₀::EmpiricalMean{T})(x::AbstractVector) where {T<:Real}
    length(x) == length(μ₀.C) || error("Wrong dimension between x and μ₀")
    return μ₀.C
end

function init_priormean_state(hyperopt_state, μ₀::EmpiricalMean)
    μ₀_state = (; C=Optimisers.init(μ₀.opt, μ₀.C))
    return merge(hyperopt_state, (; μ₀_state))
end

function update!(μ₀::EmpiricalMean{T}, hyperopt_state, grad) where {T<:Real}
    μ₀_state = hyperopt_state.μ₀_state
    C, ΔC = Optimisers.apply(μ₀.opt, μ₀_state.C, μ₀.C, grad.C)
    μ₀.C .+= ΔC
    return merge(hyperopt_state, (; μ₀_state=(; C)))
end
