using InverseLaplace
using Random
using StatsFuns, SpecialFunctions
using ForwardDiff

struct LaplaceTransformDistribution{T,TAlg}
    f::Function # Laplace transform of the pdf
    c²::T
    alg::TAlg # Algorithm to compute the inverse Laplace transform
    function LaplaceTransformDistribution{T,TAlg}(f::Function,c²::T=0.0,alg::TAlg=BromwichInverseLaplace()) where {T<:Real,TAlg}
        @assert _check_f(f) "The function passed is not valid"# Do series of check on f
        @assert c²>0 "c² has to a be non-negative real"
        new{T,TAlg}(f,c²,alg)
    end
end

function _check_f(f)
    return true
end
_gradf(d::LaplaceTransformDistribution,x::Real) = ForwardDiff.gradient(dist.f,[x])[1]
_gradlogf(d::LaplaceTransformDistribution,x::Real) = ForwardDiff.gradient(log∘dist.f,[x])[1]
_hessianlogf(d::LaplaceTransformDistribution,x::Real) = ForwardDiff.hessian(log∘dist.f,[x])[1]

Distributions.pdf(dist::LaplaceTransformDistribution,x::Real) = apply_f(dist,x)
Distributions.mean(dist::LaplaceTransformDistribution) = _gradf(dist,dist.c²)/dist.f(dist.c²)
Distributions.var(dist::LaplaceTransformDistribution) = _hessianlogf(dist,dist.c²)/dist.f(dist.c²)-mean(dist)^2

struct BromwichInverseLaplace{T}
    A::Int
    l::Int
    m::Int
    n::Int
    expA2l::T
    ijπl::Vector{Complex{T}}
    expijπl::Vector{Complex{T}}
    coefs::Vector{T}
    altern::Vector{Int}
    b::Vector{T}
    function BromwichInverseLaplace(A::Int=19,l::Int=1,m::Int=11,n::Int=28)
        @assert A>0 "A should be a positive integer"
        @assert l>0 "l should be a positive integer"
        @assert m>0 "m should be a positive integer"
        @assert n>0 "n should be a positive integer"
        T = typeof(exp(A/(2*l)))
        ijπl = 1im.*(1:l).*π/l
        new{T}(A,l,m,n,exp(A/(2*l)),ijπl,exp.(ijπl),binomial.(m,0:m)./(2^m),(-1).^(0:(n+m)),zeros(n+m),zeros())
    end
end

function Random.rand(dist::LaplaceTransformDistribution,n::Int)
    nTries = 100
    i = 0
    while i < nTries
        try
            if i > 10
                @warn "This is taking more than 10 tries! Trie $i/$nTries!"
            end
            return laptrans(dist,n=n)
        catch e
            if e isa AssertionError
                @warn "$(e.msg), Trying again"
                i+=1
                continue
            else
                rethrow(e)
            end
        end
    end
    @error "Sampler failed to converge"
end

@inline function apply_f(dist::LaplaceTransformDistribution,t)
    if t==0
        return zero(t)
    else
        invlaplace(dist.f,t,dist.alg)
    end
end

@inline function apply_F(dist::LaplaceTransformDistribution,t)
    #Take care of the spe
    if t == 0
        return zero(t)
    else
        invlaplace(s -> dist.f(s) / s,t,dist.alg)
    end
end

function invlaplace(f::Function,t::Real,A::Int=19,l::Int=1,m::Int=11,n::Int=28)
    # n is the burnin, m is the number of components, A and l aare discretization parameters
    alg = BromwichInverseLaplace(A,l,m,n)
    invlaplace(f,t,alg)
end

function invlaplace(f::Function,t::Real,alg::BromwichInverseLaplace)
    scaled_t = 1/(2*alg.l*t)
    alg.b[1] = f(alg.A*scaled_t) + 2*sum(real.(f.(alg.A*scaled_t .+ alg.ijπl./t).*alg.expijπl))
    @inbounds for k in 1:(alg.n+alg.m)
        alg.b[k+1] = 2*sum(real.(f.(alg.A*scaled_t .+ alg.ijπl./t .+ 1im*π*k/t).*alg.expijπl))
    end
    acoeff = alg.eA2l*scaled_t
    cumsum!(alg.s,acoeff.*alg.b.*alg.altern)
    return dot(view(alg.s,(alg.n+1):end),alg.coeffs)
end

# Laplace transform implemented from "Computational Probability Grassmann 2000"
function laptrans(dist::LaplaceTransformSampler;n::Int=10,jmax::Int=500,kmax::Int=1000,b::Real=2.0,τ::Real=1e-7)
    # Step 1
    global u = sort!(rand(n))
    # Step 2
    global xmax = 0.1
    j = 0
    while apply_F(dist,xmax) < u[end] && j < jmax
        xmax = b*xmax
        j += 1
    end
    @assert j!=jmax "Xmax search failed to converge"
    x = zeros(n)
    # Step 3
    for i in 1:n
        x_L  = i == 1 ? 0.0 : x[i-1]; x_U = xmax; k = 0; t = x_L;
        while abs(apply_F(dist,t)-u[i])>τ && k <kmax
            k += 1
            t = t - (apply_F(dist,t)-u[i])/apply_f(dist,t)
            if t < x_L || t > x_U
                t = 0.5*(x_L+x_U)
            end
            if apply_F(dist,t) <= u[i]
                x_L = t
            else
                x_U = t
            end
        end
        @assert k != kmax "Modified Newton-Raphson procedure failed to converge for i=$i, x[$i]=$t"
        x[i] = t
    end
    # Step 4
    return shuffle!(x)
end

ν=3.0

testfunction(x) = (1+x/ν)^(-0.5*(ν+1))
testfunction(x) = sech(sqrt(x/2))
testfunction(x) = (1+sqrt(3*x))*exp(-sqrt(3*x))
testfunction(x) = (1+sqrt(5*x)+5/3*x)*exp(-sqrt(5*x))
testdist = LaplaceTransformSampler(testfunction,nothing)

laptrans(testdist,n=1)[1]
using Plots
using AugmentedGaussianProcesses
p = histogram(rand(testdist,10000),bins=range(0,3,length=100),normalize=true)|>display
using Distributions
histogram!(rand(Gamma((ν+1)/2,1/ν),10000),alpha=0.5,bins=range(0,10,length=100),normalize=true)
histogram!([AugmentedGaussianProcesses.draw(AugmentedGaussianProcesses.PolyaGammaDist(),1,0) for i in 1:10000],alpha=0.5,bins=range(0,10,length=100),normalize=true)
