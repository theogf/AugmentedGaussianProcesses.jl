
### Global constant allowing to chose between forward_diff and zygote_diff for hyperparameter optimization ###
const ADBACKEND = Ref(:zygote)

function setadbackend(ad_backend::Symbol)
    (ad_backend == :ForwardDiff || ad_backend == :Zygote) ||
        error("Wrong AD Backend symbol, options are :ForwardDiff or :Zygote")
    return ADBACKEND[] = ad_backend
end

opt(::AbstractInducingPoints) = nothing
opt(Z::OptimIP) = Z.opt
data(Z::OptimIP) = Z.Z
data(Z::AbstractInducingPoints) = Z

## Generic fallback when gradient is nothing
update!(::Any, ::Any, ::Nothing) = nothing

## Generic fallback when optimizer is nothing
update!(::Nothing, ::Kernel, ::NamedTuple) = nothing
update!(::Nothing, ::AbstractInducingPoints, ::AbstractArray) = nothing

## Updating prior mean parameters ##
function update!(μ::PriorMean, g::AbstractVector, X::AbstractVector)
    return update!(μ, g, X)
end

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
function update!(opt, k::Union{Kernel,Transform}, g::NamedTuple)
    foreach(pairs(g)) do (fieldname, grad)
        update!(opt, getfield(k, fieldname), grad)
    end
end

function update!(opt, x::AbstractArray, g::AbstractArray)
    Δ = Optimise.apply!(opt, x, x .* g)
    @. x = exp(log(x) + Δ) # Always assume that parameters need to be positive
end

## Updating inducing points
function update!(opt, Z::AbstractInducingPoints, Z_grads::AbstractArray)
    for (z, zgrad) in zip(Z, Z_grads)
        z .+= Optimise.apply!(opt, z, zgrad)
    end
end
