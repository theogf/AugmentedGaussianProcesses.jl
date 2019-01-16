# File for sampling from a Polya Gamma Distribution
module PGSampler

using Distributions
__TRUNC = 0.64;
__TRUNC_RECIP = 1.0 / __TRUNC;
export PolyaGammaDist
mutable struct PolyaGammaDist{T}
  # For sum of Gammas.
  T::Int64;
  bvec::Vector{T}

  # Draw functions
  draw::Function
  draw_sum_of_gammas::Function
  draw_like_devroye::Function

  # Utility.
  set_trunc::Function

  # Helper.
  #Constructor
  function PolyaGammaDist{T}(;trunc = 200) where T
    this = new{T}()
    this.set_trunc = function(trunc)
      set_trunc(this,trunc)
    end
    this.set_trunc(trunc);
    this.draw = function(n,z)
      draw(n,z)
    end
    this.draw_sum_of_gammas = function(n,z)
      draw_sum_of_gammas(n,z)
    end
    this.draw_like_devroye = function(Z)
      draw_like_devroye(Z)
    end
    return this
  end
end

function PolyaGammaDist(;trunc=200)
	PolyaGammaDist{Float64}()
end
# ////////////////////////////////////////////////////////////////////////////////
# 				 // Utility //
# ////////////////////////////////////////////////////////////////////////////////

function set_trunc(pg::PolyaGammaDist,trunc::Int64)

  if trunc < 1
    @warn "PolyaGamma(int trunc): trunc < 1. Set trunc=1."
    trunc = 1;
  end

  pg.T = trunc;
  pg.bvec = zeros(pg.T);

  for k in 1:pg.T
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

function pigauss(x::T, Z::T) where {T<:Real}
  #Z = 1/μ, λ= 1.0
  b = sqrt(1.0 / x) * (x * Z - 1);
  a = -sqrt(1.0 / x) * (x * Z + 1);
  y = cdf(Distributions.Normal(),b) + exp(2 * Z) * cdf(Distributions.Normal,a);
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
  X = t + 1.0;
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
      X = 1 + E1 * t;
      X = t / (X * X);
      alpha = exp(-0.5 * Z*Z * X);
    end
  else
    mu = 1.0 / Z;
    while (X > t)
      Y = rand(Normal); Y = Y^2;
      half_mu = 0.5 * mu;
      mu_Y    = mu  * Y;
      X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      if rand(Unif) > mu / (mu + X)
	       X = mu*mu / X;
       end
    end
  end
  return X;
end

# ////////////////////////////////////////////////////////////////////////////////
# 				  // Sample //
# ////////////////////////////////////////////////////////////////////////////////


function draw(n::T, z::T) where {T<:Real}
  if n < 1
    @warn "PolyaGamma::draw: n < 1.  Set n = 1."
    n = 1;
  end
  sum = 0.0;
  for i in 1:n
    sum = sum + draw_like_devroye(z);
  end
  return sum;
end # draw

function draw_sum_of_gammas(n::T, z::T, pg::PolyaGammaDist) where {T<:Real}
  x = 0.0;
  kappa = z * z;
  Gam = Gamma(n,1.0);
  for k in 1:pg.T
    x += rand(Gam) / (bvec[k] + kappa);
  end
  return 2.0 * x;
end

function draw_like_devroye(Z::T) where {T<:Real}
  # Change the parameter.
  Z = abs(Z) * 0.5;

  # Now sample 0.25 * J^*(1, Z := Z/2).
  fz = 0.125 * pi*pi + 0.5 * Z*Z;
  # ... Problems with large Z?  Try using q_over_p.
  # double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
  # double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);

  X = 0.0;
  S = 1.0;
  Y = 0.0;
  # int iter = 0; If you want to keep track of iterations.
  Unif = Uniform()
  Expo = Exponential(1.0/1.0)
  while (true)

    if rand(Unif) < mass_texpon(Z)
      X = __TRUNC + rand(Expo) / fz;
    else
      X = rtigauss(Z);
    end
    S = a(0, X);
    Y = rand(Unif) * S;
    n = 0;
    go = true;

    # Cap the number of iterations?
    while (go)
      n = n+1;
      if n%2==1
	       S = S - a(n, X);
	       if Y<=S; return 0.25 * X;end;
      else
	       S = S + a(n, X);
	       if Y>S; go = false; end;
      end

    end
    # Need Y <= S in event that Y = S, e.g. when X = 0.

  end
end # draw_like_devroye

# ////////////////////////////////////////////////////////////////////////////////
# 			      // Static Members //
# ////////////////////////////////////////////////////////////////////////////////

function jj_m1(b::T, z::T) where T<:Real
    z = abs(z);
    m1 = 0.0;
    if z > 1e-12
	     m1 = b * tanh(z) / z;
    else
	     m1 = b * (1 - (1.0/3) * pow(z,2) + (2.0/15) * pow(z,4) - (17.0/315) * pow(z,6));
    end
    return m1;
end

function jj_m2(b::T, z::T) where {T<:Real}
    z = abs(z);
    m2 = 0.0;
    if (z > 1e-12)
	     m2 = (b+1) * b * pow(tanh(z)/z,2) + b * ((tanh(z)-z)/pow(z,3));
    else
       m2 = (b+1) * b * pow(1 - (1.0/3) * pow(z,2) + (2.0/15) * pow(z,4) - (17.0/315) * pow(z,6), 2) +
	          b * ((-1.0/3) + (2.0/15) * pow(z,2) - (17.0/315) * pow(z,4));
    end
    return m2;
end

function pg_m1(b::T, z::T) where {T<:Real}
    return jj_m1(b, 0.5 * z) * 0.25;
end

function pg_m2(b::T, z::T) where {T<:Real}
    return jj_m2(b, 0.5 * z) * 0.0625;
end

end #module PGSampler
