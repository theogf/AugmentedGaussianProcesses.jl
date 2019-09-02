"""Module for a Generalized Inverse Gaussian Sampler"""
module GIGSampler

using Distributions
using SpecialFunctions

export GeneralizedInverseGaussian

"""Sampler object"""
struct GeneralizedInverseGaussian{T<:Real} <: Distributions.ContinuousUnivariateDistribution
	a::T
    b::T
    p::T
    function GeneralizedInverseGaussian{T}(a::T, b::T, p::T) where T
        Distributions.@check_args(GeneralizedInverseGaussian, a > zero(a) && b > zero(b))
        new{T}(a, b, p)
    end
end

function GeneralizedInverseGaussian(a::T, b::T, p::T) where T
	GeneralizedInverseGaussian{T}(a::T, b::T, p::T)
end

Distributions.params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
@inline Distributions.partype(d::GeneralizedInverseGaussian{T}) where T <: Real = T


function Distributions.mean(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    q = sqrt(a * b)
    (sqrt(b) * besselk(p + 1, q)) / (sqrt(a) * besselk(p, q))
end

function Distributions.var(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    q = sqrt(a * b)
    r = besselk(p, q)
    (b / a) * ((besselk(p + 2, q) / r) - (besselk(p + 1, q) / r)^2)
end

Distributions.mode(d::GeneralizedInverseGaussian) = ((d.p - 1) + sqrt((d.p - 1)^2 + d.a * d.b)) / d.a


function Distributions.pdf(d::GeneralizedInverseGaussian{T}, x::Real) where T <: Real
    if x > 0
        a, b, p = params(d)
        (((a / b)^(p / 2)) / (2 * besselk(p, sqrt(a * b)))) * (x^(p - 1)) * exp(- (a * x + b / x) / 2)
    else
        zero(T)
    end
end

function Distributions.logpdf(d::GeneralizedInverseGaussian{T}, x::Real) where T <: Real
    if x > 0
        a, b, p = params(d)
        (p / 2) * (log(a) - log(b)) - log(2 * besselk(p, sqrt(a * b))) + (p - 1) * log(x) - (a * x + b / x) / 2
    else
        -T(Inf)
    end
end

function Distributions.rand(d::GeneralizedInverseGaussian)
    a, b, p = params(d)
    α = sqrt(a / b)
    β = sqrt(a * b)
    λ = abs(p)
    if (λ > 1) || (β > 1)
        x = _rou_shift(λ, β)
    elseif β >= min(0.5, (2 / 3) * sqrt(1 - λ))
        x = _rou(λ, β)
    else
        x = _hormann(λ, β)
    end
    p >= 0 ? x / α : 1 / (α * x)
end
function _gigqdf(x::Real, λ::Real, β::Real)
    (x^(λ - 1)) * exp(-β * (x + 1 / x) / 2)
end

function _hormann(λ::Real, β::Real)
    # compute bounding rectangles
    m = β / (1 - λ + sqrt((1 - λ)^2 + β^2))  # mode
    x0 = β / (1 - λ)
    xstar = max(x0, 2 / β)
    # in subdomain (0, x0)
    k1 = _gigqdf(m, λ, β)
    a1 = k1 * x0
    # in subdomain (x0, 2 / β), may be empty
    if x0 < 2 / β
        k2 = exp(-β)
        a2 = λ == 0 ? k2 * log(2 / (β^2)) : k2 * ((2 / β)^λ - x0^λ) / λ
    else
        k2 = 0
        a2 = 0
    end
    # in subdomain (xstar, Inf)
    k3 = xstar^(λ - 1)
    a3 = 2k3 * exp(-xstar * β / 2) / β
    a = a1 + a2 + a3

    # perform rejection sampling
    while true
        u = rand()
        v = a * rand()
        if v <= a1
            # in subdomain (0, x0)
            x = x0 * v / a1
            h = k1
        elseif v <= a1 + a2
            # in subdomain (x0, 2 / β)
            v -= a1
            x = λ == 0 ? β * exp(v * exp(β)) : (x0^λ + v * λ / k2)^(1 / λ)
            h = k2 * x^(λ - 1)
        else
            # in subdomain (xstar, Inf)
            v -= a1 + a2
            x = -2log(exp(-xstar * β / 2) - v * β / (2k3)) / β
            h = k3 * exp(-x * β / 2)
        end
        if u * h <= _gigqdf(x, λ, β)
            return x
        end
    end
end

function _rou(λ::Real, β::Real)
    # compute bounding rectangle
    m = β / (1 - λ + sqrt((1 - λ)^2 + β^2))  # mode
    xpos = (1 + λ + sqrt((1 + λ)^2 + β^2)) / β
    vpos = sqrt(_gigqdf(m, λ, β))
    upos = xpos * sqrt(_gigqdf(xpos, λ, β))

    # perform rejection sampling
    while true
        u = upos * rand()
        v = vpos * rand()
        x = u / v
        if v^2 <= _gigqdf(x, λ, β)
            return x
        end
    end
end

function _rou_shift(λ::Real, β::Real)
    # compute bounding rectangle
    m = (λ - 1 + sqrt((λ - 1)^2 + β^2)) / β  # mode
    a = -2(λ + 1) / β - m
    b = 2(λ - 1) * m / β - 1
    p = b - (a^2) / 3
    q = 2(a^3) / 27 - (a * b) / 3 + m
    ϕ = acos(-(q / 2) * sqrt(-27 / (p^3)))  # Cardano's formula
    r = sqrt(-4p / 3)
    xneg = r * cos(ϕ / 3 + 4π / 3) - a / 3
    xpos = r * cos(ϕ / 3) - a / 3
    vpos = sqrt(_gigqdf(m, λ, β))
    uneg = (xneg - m) * sqrt(_gigqdf(xneg, λ, β))
    upos = (xpos - m) * sqrt(_gigqdf(xpos, λ, β))

    # perform rejection sampling
    while true
        u = (upos - uneg) * rand() + uneg
        v = vpos * rand()
        x = max(u / v + m, 0)
        if v^2 <= _gigqdf(x, λ, β)
            return x
        end
    end
end

end #module GIGSampler
