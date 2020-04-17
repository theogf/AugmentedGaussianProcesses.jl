const __TRUNC = 0.64;
const __TRUNC_RECIP = 1.0 / __TRUNC;
export PolyaGamma


struct PGSampler{T} <: Distributions.Sampleable{Univariate, Continuous}

end

"""
    PolyaGamma(b::Real, c::Real; trunc::Int = 200)

Polya-Gamma distribution
"""
struct PolyaGamma{T} <: Distributions.ContinuousUnivariateDistribution
  # For sum of Gammas.
 	b::T
 	c::T
 	trunc::Int64
	bvec::Vector{T}
end

function PolyaGamma(b::T1, c::T2; trunc::Int = 200) where {T1<:Real, T2<:Real}
	T = promote_type(T1, T2)
	T = T isa Int ? Float64 : T
	@assert b >= 0
	trunc = if trunc < 1
	  	@warn "PolyaGamma(trunc::Int): trunc < 1. Set trunc=1."
	  	1
    else
		trunc
	end
	bvec = zeros(T, trunc)
	for i in 1:trunc
	    bvec[i] = 4* π^2 * (i - 0.5)^2;
	end
	PolyaGamma{T}(b, c, trunc, bvec)
end

Distributions.params(d::PolyaGamma) = (d.b, d.c)
Distributions.minimum(::PolyaGamma{T}) where {T} = zero(T)
Distributions.maximum(::PolyaGamma) = Inf

function Distributions.pdf(d::PolyaGamma, x::Real, nmax::Int = 10)
    cosh(d.c / 2)^d.b * 2.0^(d.b - 1) / gamma(d.b) * sum(
        ((-1)^n) * gamma(n + d.b) / gamma(n + 1) * (2 * n + d.b) /
        (sqrt(2π * x^3)) * exp(-(2 * n + d.b)^2 / (8 * x) - d.c^2 / 2 * x)
        for n = 0:nmax
    )
end

Distributions.mean(d::PolyaGamma) = 0.5 * d.b / d.c * tanh( 0.5 * d.c)

# sample from PG(b,c)
function Distributions.rand(d::PolyaGamma)
  if d.b == 0
    # @warn "draw(PolyaGamma): n < 1.  Set n = 1." maxlog=1
    return 0.0
  end
  sum = 0.0;
  for i in 1:d.b
    sum = sum + draw_like_devroye(d.c);
  end
  return sum;
end # draw

## Utility

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
