mutable struct ConstantMean{T<:Real,O} <: PriorMean{T}
    C::Vector{T}
    opt::O
end

"""
    ConstantMean(c::Real = 1.0; opt=ADAM(0.01))

## Arguments
- `c::Real` : Constant value

Construct a prior mean with constant `c`
Optionally set an optimiser `opt` (`ADAM(0.01)` by default)
"""
function ConstantMean(c::T=1.0; opt=ADAM(0.01)) where {T<:Real}
    return ConstantMean{T,typeof(opt)}([c], opt)
end

function Base.show(io::IO, ::MIME"text/plain", μ₀::ConstantMean)
    return print(io, "Constant Mean Prior (c = ", first(μ₀.C), ")")
end

(μ::ConstantMean{T})(::Real) where {T<:Real} = first(μ.C)
(μ::ConstantMean{T})(x::AbstractMatrix) where {T<:Real} = fill(first(μ.C), size(x, 1))

function init_priormean_state(hyperopt_state, μ₀::ConstantMean)
    μ₀_state = (; C=init(μ₀.opt, μ₀.C))
    return merge(hyperopt_state, (;μ₀_state))
end

function update!(μ₀::ConstantMean{T}, hyperopt_state, grad) where {T<:Real}
    μ₀_state = hyperopt_state.μ₀_state
    C, ΔC = Optimisers.apply(μ₀.opt, μ₀_state.C, μ₀.w, grad)
    μ₀.C .+= ΔC
    return merge(hyperopt_state, (;μ₀_state=(;C)))
end
