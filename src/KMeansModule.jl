#File for the Assumption Free K MC2 algorithm (KMeans)
module KMeansModule

using Distributions
using StatsBase
using Clustering
using OMGP.KernelFunctions

export KMeansInducingPoints
export KMeansAlg, StreamOnline, Webscale, CircleKMeans
export total_cost
export init!, update!

function find_nearest_center(X,C,kernel=0)
    nC = size(C,1)
    best = Int64(1); best_val = Inf
    for i in 1:nC
        val = distance(X,C[i,:],kernel)
        if val < best_val
            best_val = val
            best = i
        end
    end
    return best,best_val
end

function total_cost(X,C,kernel)
    n = size(X,1)
    tot = 0
    for i in 1:n
        tot += find_nearest_center(X[i,:],C,kernel)[2]
    end
    return tot
end

function distance(X,C,kernel=0)
    if kernel == 0
        return norm(X-C,2)^2
    else
        return compute(kernel,X,X)+compute(kernel,C,C)-2*compute(kernel,X,C)
    end
end

abstract type KMeansAlg end;

function total_cost(X::Array{Float64,2},alg::KMeansAlg,kernel=0)
    n = size(X,1)
    tot = 0
    for i in 1:n
        tot += find_nearest_center(X[i,:],alg.centers,kernel)[2]
    end
    return tot
end

mutable struct OfflineKmeans <: KMeansAlg
    kernel::Kernel
    k::Int64
    centers::Array{Float64,2}
    function OfflineKmeans()
        return new()
    end
end

function init!(alg::OfflineKmeans,X,model,k::Int64)
    @assert size(X,1)>=k "Input data not big enough given $k"
    alg.k = k
end

function update!(alg::OfflineKmeans,X,model)
    results = kmeans(X',alg.k)
    alg.centers = results.centers'
    return results
end

mutable struct Webscale <: KMeansAlg
    k::Int64
    v::Array{Int64,1}
    centers::Array{Float64,2}
    function Webscale()
        return new()
    end
end


function init!(alg::Webscale,X,model,k::Int64)
    @assert size(X,1)>=k "Input data not big enough given $k"
    alg.k = k;
    alg.v = zeros(Int64,k);
    alg.centers = X[sample(1:size(X,1),k),:];
end

function update!(alg::Webscale,X,model)
    b = size(X,1)
    d = zeros(Int64,b)
    for i in 1:b
        d[i] = find_nearest_center(X[i,:],alg.centers)[1]
    end
    for i in 1:b
        alg.v[d[i]] += 1
        η = 1/alg.v[d[i]]
        alg.centers[d[i],:] = (1-η)*alg.centers[d[i],:]+ η*X[i,:]
    end
end

mutable struct StreamOnline <: KMeansAlg
    k_target::Int64
    k_efficient::Int64
    k::Int64
    f::Float64
    q::Int64
    centers::Array{Float64,2}
    function StreamOnline()
        return new()
    end
end


function init!(alg::StreamOnline,X,model,k::Int64)
    @assert size(X,1)>=10 "The first batch of data should be bigger than 10 samples"
    alg.k_target = k;
    alg.k_efficient = max(1,ceil(Int64,(k-15)/5))
    if alg.k_efficient+10 > size(X,1)
         alg.k_efficient = 0
    end
    alg.centers = X[sample(1:size(X,1),alg.k_efficient+10,replace=false),:]
    # alg.centers = X[1:(alg.k_efficient+10),:]
    alg.k = alg.k_efficient+10
    w=zeros(alg.k)
    for i in 1:alg.k
        w[i] = 0.5*find_nearest_center(alg.centers[i,:],alg.centers[1:alg.k.!=i,:])[2]
        # w[i] = 0.5*find_nearest_center(X[i,:],alg.centers[1:alg.k.!=i,:])[2]
    end
    alg.f = sum(sort(w)[1:10]) #Take the 10 smallest values
    alg.q = 0
end

function update!(alg::StreamOnline,X,model)
    b = size(X,1)
    new_centers = Matrix(0,size(X,2))
    for i in 1:b
        val = find_nearest_center(X[i,:],alg.centers)[2]
        if val>(alg.f*rand())
            # new_centers = vcat(new_centers,X[i,:]')
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.q += 1
            alg.k += 1
        end
        if alg.q >= alg.k_efficient
            alg.q = 0
            alg.f *=10
        end
    end
    # alg.centers = vcat(alg.centers,new_centers)
end



mutable struct CircleKMeans <: KMeansAlg
    lim::Float64
    k::Int64
    centers::Array{Float64,2}
    function CircleKMeans(;lim=0.9)
        return new(lim)
    end
end


function init!(alg::CircleKMeans,X,model,k::Int64;lim=0.9)
    @assert lim < 1.0 && lim > 0 "lim should be between 0 and 1"
    @assert size(X,1)>=10 "The first batch of data should be bigger than 10 samples"
    alg.centers = reshape(X[1,:],1,size(X,2))
    alg.k = 1
    update!(alg,X[2:end,:],model)
end

function update!(alg::CircleKMeans,X,model)
    b = size(X,1)
    for i in 1:b
        d = find_nearest_center(X[i,:],alg.centers,model.kernel)[2]
        if d>2*(1-alg.lim)
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.k += 1
        end
    end
end



#-----------------------------------------------------#


#Return K inducing points from X, m being the number of Markov iterations for the seeding
function KMeansInducingPoints(X,K,m;weights=0)
    C = (KmeansSeed(X,K,m))'
    if weights!=0
        kmeans!(X',C,weights=weights)
    else
        kmeans!(X',C)
    end
return C'
end
#Fast and efficient seeding for KMeans
function KmeansSeed(X,K,m) #X is the data, K the number of centers wanted, m the number of Markov iterations
  N = size(X,1)
  #Preprocessing, sample first random center
  init = StatsBase.sample(1:N,1)
  C = zeros(K,size(X,2))
  C[1,:] = X[init,:]
  q = zeros(N)
  for i in 1:N
    q[i] = 0.5*norm(X[i,:]-C[1])^2
  end
  sumq = sum(q)
  q = Weights(q/sumq + 1.0/(2*N),1)
  uniform = Distributions.Uniform(0,1)
  for i in 2:K
    x = X[StatsBase.sample(1:N,q,1),:] #weighted sampling,
    mindist = mindistance(x,C,i-1)
    for j in 2:m
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

#Compute the minimum distance
function mindistance(x,C,K) #Point to look for, collection of centers, number of centers computed
  mindist = Inf
  for i in 1:K
    mindist = min.(norm(x-C[i])^2,mindist)
  end
  return mindist
end

end #module
