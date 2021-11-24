using Distributions, Random
using Statistics
using SpecialFunctions
const __TRUNC = 0.64;
const __TRUNC_RECIP = 1.0 / __TRUNC;
"""
    PolyaGamma(b::Int, c::Real)

## Arguments
- `b::Int` 
- `c::Real` exponential tilting

## Keyword Arguments
- `nmax::Int` Sampling parameter
- `trunc::Int` Sampling parameter

Create a PolyaGamma sampler with parameters `b` and `c`
"""
struct PolyaGamma{Tc,A} <: Distributions.ContinuousUnivariateDistribution
    # For sum of Gammas.
    b::Int
    c::Tc
    trunc::Int
    nmax::Int
    bvec::A
    #Constructor
    function PolyaGamma{T}(b::Int, c::T, trunc::Int, nmax::Int) where {T<:Real}
        if trunc < 1
            @warn "trunc < 1. Setting trunc=1."
            trunc = 1
        end
        bvec = [convert(T, (twoπ * (k - 0.5))^2) for k in 1:trunc]
        return new{typeof(c),typeof(bvec)}(b, c, trunc, nmax, bvec)
    end
end

Statistics.mean(d::PolyaGamma) = d.b / (2 * d.c) * tanh(d.c / 2)

function PolyaGamma(b::Int, c::T; nmax::Int=10, trunc::Int=200) where {T<:Real}
    return PolyaGamma{T}(b, c, trunc, nmax)
end

function Distributions.pdf(d::PolyaGamma, x)
    return cosh(d.c / 2)^d.b * 2.0^(d.b - 1) / gamma(d.b) * sum(
        ((-1)^n) * gamma(n + d.b) / gamma(n + 1) * (2 * n + b) / (sqrt(2 * π * x^3)) *
        exp(-(2 * n + b)^2 / (8 * x) - c^2 / 2 * x) for n in 0:(d.nmax)
    )
end

## Sampling
function Distributions.rand(rng::AbstractRNG, d::PolyaGamma{T}) where {T<:Real}
    if iszero(d.b)
        return zero(T)
    end
    return sum(Base.Fix1(draw_like_devroye, rng), d.c * ones(d.b))
end

## Utility functions
function a(n::Int, x::Real)
    k = (n + 0.5) * π
    if x > __TRUNC
        return k * exp(-k^2 * x / 2)
    elseif x > 0
        expnt = -1.5 * (log(halfπ) + log(x)) + log(k) - 2 * (n + 1//2)^2 / x
        return exp(expnt)
    end
end

function mass_texpon(z::Real)
    t = __TRUNC

    fz = 0.125 * π^2 + z^2 / 2
    b = sqrt(1.0 / t) * (t * z - 1)
    a = sqrt(1.0 / t) * (t * z + 1) * -1.0

    x0 = log(fz) + fz * t
    xb = x0 - z + logcdf(Distributions.Normal(), b)
    xa = x0 + z + logcdf(Distributions.Normal(), a)

    qdivp = 4 / π * (exp(xb) + exp(xa))

    return 1.0 / (1.0 + qdivp)
end

function rtigauss(rng::AbstractRNG, z::Real)
    z = abs(z)
    t = __TRUNC
    x = t + 1.0
    if __TRUNC_RECIP > z
        alpha = 0.0
        rate = 1.0
        d_exp = Exponential(1.0 / rate)
        while (rand(rng) > alpha)
            e1 = rand(rng, d_exp)
            e2 = rand(rng, d_exp)
            while e1^2 > 2 * e2 / t
                e1 = rand(rng, d_exp)
                e2 = rand(rng, d_exp)
            end
            x = 1 + e1 * t
            x = t / x^2
            alpha = exp(-z^2 * x / 2)
        end
    else
        mu = 1.0 / z
        while (x > t)
            y = randn(rng)^2
            half_mu = mu / 2
            mu_Y = mu * y
            x = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y^2)
            if rand(rng) > mu / (mu + x)
                x = mu^2 / x
            end
        end
    end
    return x
end

# ////////////////////////////////////////////////////////////////////////////////
# 				  // Sample //
# ////////////////////////////////////////////////////////////////////////////////

function draw_like_devroye(rng::AbstractRNG, c::Real)
    # Change the parameter.
    c = abs(c) / 2

    # Now sample 0.25 * J^*(1, Z := Z/2).
    fz = 0.125 * π^2 + c^2 / 2
    # ... Problems with large Z?  Try using q_over_p.
    # double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
    # double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);

    x = 0.0
    s = 1.0
    y = 0.0
    # int iter = 0; If you want to keep track of iterations.
    d_exp = Exponential()
    while true
        if rand(rng) < mass_texpon(c)
            x = __TRUNC + rand(rng, d_exp) / fz
        else
            x = rtigauss(rng, c)
        end
        s = a(0, x)
        y = rand(rng) * s
        n = 0
        go = true

        # Cap the number of iterations?
        while (go)
            n = n + 1
            if isodd(n)
                s = s - a(n, x)
                if y <= s
                    return 0.25 * x
                end
            else
                s = s + a(n, x)
                if y > s
                    go = false
                end
            end
        end
        # Need Y <= S in event that Y = S, e.g. when x = 0.
    end
end # draw_like_devroye
