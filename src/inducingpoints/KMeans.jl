mutable struct Kmeans{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    k::Int64
    opt::O
    nMarkov::Int64
    Z::M
    function Kmeans(nInducingPoints::Integer,opt=Flux.ADAM(0.001);nMarkov=10)
        return new{Float64,Matrix{Float64},typeof(opt)}(nInducingPoints,opt,nMarkov)
    end
end

function init!(alg::Kmeans,X,y,kernel;tol=1e-3)
    @assert size(X,1)>=alg.k "Input data not big enough given $k"
    alg.Z = kmeans_ip(X,alg.k,nMarkov=alg.nMarkov,tol=tol)
end

"""Fast and efficient seeding for KMeans based on [`Fast and Provably Good Seeding for k-Means](https://las.inf.ethz.ch/files/bachem16fast.pdf)"""
function kmeans_seeding(X::AbstractArray{T,N},nC::Integer,nMarkov::Integer) where {T,N} #X is the data, nC the number of centers wanted, m the number of Markov iterations
  NSamples = size(X,1)
  #Preprocessing, sample first random center
  init = StatsBase.sample(1:NSamples,1)
  C = zeros(nC,size(X,2))
  C[1,:] = X[init,:]
  q = zeros(NSamples)
  for i in 1:NSamples
    q[i] = 0.5*norm(X[i,:].-C[1])^2
  end
  sumq = sum(q)
  q = Weights(q/sumq .+ 1.0/(2*NSamples),1)
  uniform = Distributions.Uniform(0,1)
  for i in 2:nC
    x = X[StatsBase.sample(1:NSamples,q,1),:] #weighted sampling,
    mindist = mindistance(x,C,i-1)
    for j in 2:nMarkov
      y = X[StatsBase.sample(q),:] #weighted sampling
      dist = mindistance(y,C,i-1)
      if (dist/mindist > rand(uniform))
        x = y;  mindist = dist
      end
    end
    C[i,:]=x
  end
  return C
end

#Return K inducing points from X, m being the number of Markov iterations for the seeding
function kmeans_ip(X::AbstractArray{T,N},nC::Integer;nMarkov::Integer=10,kweights::Vector{T}=[0.0],tol=1e-3) where {T,N}
    C = copy(transpose(kmeans_seeding(X,nC,nMarkov)))
    if kweights!=[0.0]
        Clustering.kmeans!(copy(transpose(X)),C,weights=kweights,tol=tol)
    else
        Clustering.kmeans!(copy(transpose(X)),C)
    end
    return copy(transpose(C))
end
