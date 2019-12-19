"""Module for a Polya-Gamma Sampler"""
module PGSampler

using Distributions
using SpecialFunctions
const __TRUNC = 0.64;
const __TRUNC_RECIP = 1.0 / __TRUNC;
export PolyaGammaDist, draw
"""Sampler object"""
mutable struct PolyaGammaDist{T}
  # For sum of Gammas.
  trunc::Int64;
  bvec::Vector{T}
  # Helper.
  #Constructor
  function PolyaGammaDist{T}(trunc::Int) where {T<:Real}
    this = new{T}()
	set_trunc(this,trunc)
	return this
  end
end

function PolyaGammaDist(;trunc::Int=200)
	PolyaGammaDist{Float64}(trunc)
end


function Distributions.pdf(d::PolyaGammaDist,x,b,c,nmax::Int=10)
	cosh(c/2)^b*2.0^(b-1)/gamma(b)*sum(((-1)^n)*gamma(n+b)/gamma(n+1)*(2*n+b)/(sqrt(2*pi*x^3))*exp(-(2*n+b)^2/(8*x)-c^2/2*x) for n in 0:nmax)
end
## Utility

function set_trunc(pg::PolyaGammaDist{T},trunc::Int64) where {T<:Real}

  if trunc < 1
    @warn "PolyaGamma(trunc::Int): trunc < 1. Set trunc=1."
    trunc = 1;
  end

  pg.trunc = trunc;
  pg.bvec = zeros(T,pg.trunc);

  for k in 1:pg.trunc
    d = k - 0.5;
    pg.bvec[k] = 4* pi^2 * d * d;
  end
end # set_trunc

function a(n::Int64, x::T) where {T<:Real}
  K = (n + 0.5) * pi;
  y = 0;
  if x > __TRUNC
    y = K * exp( -0.5 * K*K * x );
  elseif x > 0
    expnt = -1.5 * (log(0.5 * pi)  + log(x)) + log(K) - 2.0 * (n+0.5)*(n+0.5) / x;
    y = exp(expnt);
  end
  return y;
end

function mass_texpon(Z::T) where {T<:Real}
  t = __TRUNC;

  fz = 0.125 * pi*pi + 0.5 * Z^2;
  b = sqrt(1.0 / t) * (t * Z - 1);
  a = sqrt(1.0 / t) * (t * Z + 1) * -1.0;

  x0 = log(fz) + fz * t;
  xb = x0 - Z + logcdf(Distributions.Normal(),b);
  xa = x0 + Z + logcdf(Distributions.Normal(),a);

  qdivp = 4 / pi * ( exp(xb) + exp(xa) );

  return 1.0 / (1.0 + qdivp);
end # mass_texpon

function rtigauss(Z::T) where {T<:Real}
  Z = abs(Z);
  t = __TRUNC;
  x = t + 1.0;
  Unif = Uniform(); # between 0 and 1
  Normal = Distributions.Normal(0.0,1.0);
  if __TRUNC_RECIP > Z
    alpha = 0.0;
    rate = 1.0;
    Expo = Exponential(1.0/rate);
    while (rand(Unif) > alpha)
      E1 = rand(Expo); E2 = rand(Expo);
      while  E1*E1 > 2 * E2 / t
	       E1 = rand(Expo); E2 = rand(Expo);
      end
      x = 1 + E1 * t;
      x = t / (x * x);
      alpha = exp(-0.5 * Z*Z * x);
    end
  else
    mu = 1.0 / Z;
    while (x > t)
      Y = rand(Normal); Y = Y^2;
      half_mu = 0.5 * mu;
      mu_Y    = mu  * Y;
      x = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      if rand(Unif) > mu / (mu + x)
	       x = mu*mu / x;
       end
    end
  end
  return x;
end

# ////////////////////////////////////////////////////////////////////////////////
# 				  // Sample //
# ////////////////////////////////////////////////////////////////////////////////


# sample from PG(b,c)
function draw(pg::PolyaGammaDist{T},n::Real, z::Real) where {T<:Real}
  if n == 0
    # @warn "draw(PolyaGamma): n < 1.  Set n = 1." maxlog=1
    return 0.0
  end
  sum = 0.0;
  for i in 1:n
    sum = sum + draw_like_devroye(z);
  end
  return sum;
end # draw

function draw_like_devroye(Z::T) where {T<:Real}
  # Change the parameter.
  Z = abs(Z) * 0.5;

  # Now sample 0.25 * J^*(1, Z := Z/2).
  fz = 0.125 * pi*pi + 0.5 * Z*Z;
  # ... Problems with large Z?  Try using q_over_p.
  # double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
  # double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);

  x = 0.0;
  S = 1.0;
  Y = 0.0;
  # int iter = 0; If you want to keep track of iterations.
  Unif = Uniform()
  Expo = Exponential(1.0/1.0)
  while (true)

    if rand(Unif) < mass_texpon(Z)
      x = __TRUNC + rand(Expo) / fz;
    else
      x = rtigauss(Z);
    end
    S = a(0, x);
    Y = rand(Unif) * S;
    n = 0;
    go = true;

    # Cap the number of iterations?
    while (go)
      n = n+1;
      if n%2==1
	       S = S - a(n, x);
	       if Y<=S; return 0.25 * x;end;
      else
	       S = S + a(n, x);
	       if Y>S; go = false; end;
      end

    end
    # Need Y <= S in event that Y = S, e.g. when x = 0.

  end
end # draw_like_devroye

end #module PGSampler
