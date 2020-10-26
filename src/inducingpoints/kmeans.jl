"""
  KMeansIP(X::AbstractMatrix, m; obsdim = 1, nMarkov = 10, weights = nothing, tol = 1e-3)

k-Means [1] initialization on the data `X` taking `m` inducing points.
The seeding is computed via [2], `nMarkov` gives the number of MCMC steps for the seeding.
Additionally `weights` can be attributed to each data point

[1] Arthur, D. & Vassilvitskii, S. k-means++: The advantages of careful seeding. in Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms 1027–1035 (Society for Industrial and Applied Mathematics, 2007).
[2] Bachem, O., Lucic, M., Hassani, S. H. & Krause, A. Fast and Provably Good Seedings for k-Means. Advances in Neural Information Processing Systems 29 55--63 (2016) doi:10.1109/tmtt.2005.863818.
"""
struct KmeansIP{S,TZ<:AbstractVector{S}} <: AIP{S,TZ}
  k::Int
  Z::TZ
end

function KmeansIP(
  X::AbstractMatrix,
  m::Integer;
  obsdim = 1,
  nMarkov = 10,
  weights = nothing,
  tol::Real = 1e-3,
)
  size(X, obsdim) >= m || "Input data not big enough given $(m)"
  return KmeansIP(
    m,
    kmeans_ip(
      X,
      m,
      obsdim = obsdim,
      nMarkov = nMarkov,
      weights = weights,
      tol = tol,
    ),
  )
end


Base.show(io::IO, alg::KmeansIP) =
  print(io, "k-Means Selection of Inducing Points (k : $(alg.k))")

#Return K inducing points from X, m being the number of Markov iterations for the seeding
function kmeans_ip(
  X::AbstractMatrix,
  nC::Integer;
  obsdim::Int = 1,
  nMarkov::Int = 10,
  weights = nothing,
  tol = 1e-5,
)
  if obsdim == 2
    C = kmeans_seeding(X, nC, nMarkov)
    if !isnothing(weights)
      kmeans!(X, C, weights = weights, tol = tol)
    else
      kmeans!(X, C, tol = tol)
    end
    return ColVecs(C)
  elseif obsdim == 1
    C = kmeans_seeding(X', nC, nMarkov)
    if !isnothing(weights)
      kmeans!(X', C, weights = weights, tol = tol)
    else
      kmeans!(X', C, tol = tol)
    end
    return ColVecs(C)
  end
end

"""Fast and efficient seeding for KMeans based on [`Fast and Provably Good Seeding for k-Means](https://las.inf.ethz.ch/files/bachem16fast.pdf)"""
function kmeans_seeding(
  X::AbstractMatrix{T},
  nC::Integer,
  nMarkov::Integer,
) where {T} #X is the data, nC the number of centers wanted, m the number of Markov iterations
  nDim, nSamples = size(X)
  #Preprocessing, sample first random center
  init = rand(1:nSamples)
  C = zeros(T, nDim, nC)
  C[:, 1] .= X[:, init]
  q = vec(pairwise(SqEuclidean(), X, C[:, 1:1], dims = 2))
  sumq = sum(q)
  q = Weights(q / sumq .+ 1.0 / (2 * nSamples), 1)
  for i = 2:nC
    x = X[:, sample(q)] # weighted sampling,
    mindist = mindistance(x, C, i - 1)
    for j = 2:nMarkov
      y = X[:, sample(q)] #weighted sampling
      dist = mindistance(y, C, i - 1)
      if (dist / mindist > rand())
        x = y
        mindist = dist
      end
    end
    C[:, i] .= x
  end
  return C
end

#Compute the minimum distance between a vector and a collection of vectors
function mindistance(
  x::AbstractVector,
  C::AbstractMatrix,
  nC::Int
)#Point to look for, collection of centers, number of centers computed
  return minimum(evaluate(SqEuclidean(), c, x) for c in eachcol(C[:, 1:nC]))
end
