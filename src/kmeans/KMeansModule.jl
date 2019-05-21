"""
    Module for computing the Kmeans approximation for finding inducing points, it is based on the k-means++ algorithm
"""
module KMeansModule

using Distributions
using StatsBase
using LinearAlgebra
using Clustering
using AugmentedGaussianProcesses.KernelModule

export KMeansInducingPoints

#Return K inducing points from X, m being the number of Markov iterations for the seeding
function KMeansInducingPoints(X::AbstractArray{T,N},nC::Integer;nMarkov::Integer=10,kweights::Vector{<:Real}=[0.0]) where {T<:Real,N}
    C = copy(transpose(KmeansSeed(X,nC,nMarkov)))
    if kweights!=[0.0]
        Clustering.kmeans!(X',C,weights=kweights,tol=1e-3)
    else
        Clustering.kmeans!(X',C,tol=1e-3)
    end
    return copy(transpose(C))
end

"""Find good approximates seeds for the k-means++ algorithm, `X` is the data, `nC` is the number of centers and `nMarkov` is the number of Markov Chain samples before stopping. Implementation of [Fast and efficient seeding for KMeans based on [Fast and Provably Good Seeding for k-Means](https://las.inf.ethz.ch/files/bachem16fast.pdf)"""
function KmeansSeed(X::AbstractMatrix{T},nC::Integer,nMarkov::Integer) where {T<:Real} #X is the data, nC the number of centers wanted, m the number of Markov iterations
  NSamples = size(X,1)
  #Preprocessing, sample first random center
  init = rand(1:NSamples)
  C = zeros(T,nC,size(X,2))
  C[1,:] .= X[init,:]
  q = zeros(T,NSamples)
  @inbounds for i in 1:NSamples
    @views q[i] = 0.5*norm(X[i,:].-C[1])^2
  end
  sumq = sum(q)
  q = Weights(q/sumq .+ 1.0/(2*NSamples),1)
  uniform = Distributions.Uniform(0,1)
  x = zeros(T,size(X,2))
  y = similar(x)
  @inbounds for i in 2:nC
    x .= vec(X[StatsBase.sample(1:NSamples,q,1),:]) #weighted sampling,
    mindist = mindistance(x,C,i-1)
    for j in 2:nMarkov
      y .= vec(X[StatsBase.sample(q),:]) #weighted sampling
      dist = mindistance(y,C,i-1)
      if (dist/mindist > rand(uniform))
        x .= y;  mindist = dist
      end
    end
    C[i,:].=x
  end
  return C
end

"""Compute the minimum distance between a vector `x` and a collection of centers `C` given the number of computed centers `nC`"""
function mindistance(x::AbstractVector{T},C::AbstractMatrix{T},nC::Integer) where {T}#Point to look for, collection of centers, number of centers computed
  mindist = T(Inf)
  @inbounds for i in 1:nC
    mindist = min(norm(x-C[i,:])^2,mindist)
  end
  return mindist
end

end #module
