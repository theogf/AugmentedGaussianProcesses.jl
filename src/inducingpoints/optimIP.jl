"""
    OptimIP(Z, opt = ADAM(1e-3))

Inducing point object containing its own optimizer
"""
struct OptimIP{S, TZ<:AbstractVector{S}, IP<:AIP{S,TZ}, O} <: AIP{S, TZ}
    Z::IP
    opt::O
    function OptimIP(Z::AIP{S, TZ}, opt=ADAM(1e-3)) where {S, TZ}
        return new{S, TZ, typeof(Z), typeof(opt)}(Z, opt)
    end
end
