
# Global constant allowing to chose between ForwardDiff and Zygote
# for hyperparameter optimization
const ADBACKEND = Ref(:Zygote)

function setadbackend(ad_backend::Symbol)
    (ad_backend == :ForwardDiff || ad_backend == :Zygote) ||
        error("Wrong AD Backend symbol, options are :ForwardDiff or :Zygote")
    return ADBACKEND[] = ad_backend
end

# Type piracy but will be introduced in a future version
# See https://github.com/FluxML/Optimisers.jl/pull/24
Optimisers.init(::Any, ::Any) = nothing

## Generic fallback when gradient is nothing
update_kernel!(::Any, ::Any, ::Nothing, ::Any) = nothing
update_Z!(::Any, ::Any, ::Nothing, ::Any) = nothing

## Generic fallback when optimizer is nothing
update_kernel!(::Nothing, ::Kernel, ::NamedTuple, ::NamedTuple) = nothing
update_kernel!(::Nothing, ::AbstractVector, ::NamedTuple, ::NamedTuple) = nothing
update_Z!(::Nothing, ::AbstractVector, ::AbstractVector) = nothing
update_Z!(::Nothing, ::AbstractVector, ::NamedTuple) = nothing

## Updating prior mean parameters ##
# function update!(μ::PriorMean, g::AbstractVector, X::AbstractVector, state)
# return update!(μ, g, X, state)
# end

## Updating kernel parameters ##

## ForwardDiff.jl approach (with destructure())
function update!(opt, k::Kernel, Δ::AbstractVector)
    ps = params(k)
    i = 1
    for p in ps
        d = length(p)
        Δp = Δ[i:(i + d - 1)]
        Δlogp = Optimise.apply!(opt, p, p .* Δp)
        @. p = exp(log(p) + Δlogp)
        i += d
    end
end

## Zygote.jl approach with named tuple
function update_kernel!(opt, k::Union{Kernel,Transform}, g::NamedTuple, state::NamedTuple)
    return NamedTuple(
        map(keys(g)) do fieldname
            Pair(
                fieldname,
                update_kernel!(
                    opt,
                    getfield(k, fieldname),
                    getfield(g, fieldname),
                    getfield(state, fieldname),
                ),
            )
        end,
    )
end

function update_kernel!(opt, x::AbstractArray, g::AbstractArray, state)
    state, Δ = Optimisers.apply(opt, state, x, x .* g)
    @. x = exp(log(x) + Δ) # Always assume that parameters need to be positive
    return state
end

## Updating inducing points
function update_Z!(opt, Z::AbstractVector, Z_grads::AbstractVector, state)
    return map(Z, Z_grads, state) do z, zgrad, st
        st, ΔZ = Optimisers.apply(opt, st, z, zgrad)
        z .+= ΔZ
        return st
    end
end

function update_Z!(opt, Z::Union{ColVecs,RowVecs}, Z_grads::NamedTuple, state)
    st, Δ = Optimisers.apply(opt, state, Z.X, Z_grads.X)
    Z.X .+= Δ
    return st
end
