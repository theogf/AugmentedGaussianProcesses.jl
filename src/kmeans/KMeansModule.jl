"""
    Module for computing the Kmeans approximation for finding inducing points, it is based on the k-means++ algorithm
"""
# module KMeansModule

using Distributions
using StatsBase
using LinearAlgebra, Clustering, Distances
using AugmentedGaussianProcesses.KernelModule

export KMeansInducingPoints
export KMeansAlg, StreamOnline, Webscale, CircleKMeans, DataSelection, OfflineKmeans
export total_cost
export init!, update!

"Abstract type for kmeans algorithm"
abstract type KMeansAlg end;
"Find the closest center to X among C, return the index and the distance"
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

"Return the total cost of the current dataset given a set of centers"
function total_cost(X,C,kernel)
    n = size(X,1)
    tot = 0
    for i in 1:n
        tot += find_nearest_center(X[i,:],C,kernel)[2]
    end
    return tot
end

"Return the total cost of the current algorithm"
function total_cost(X::Array{Float64,2},alg::KMeansAlg,kernel=0)
    n = size(X,1)
    tot = 0
    for i in 1:n
        tot += find_nearest_center(X[i,:],alg.centers,kernel)[2]
    end
    return tot
end

"Compute the distance (kernel if included) between a point and a find_nearest_center"
function distance(X,C,kernel=0)
    if kernel == 0
        return norm(X-C,2)^2
    else
        c = KernelModule.kappa(kernel)
        return c(evaluate(getmetric(kernel),X,X))+c(evaluate(getmetric(kernel),C,C))-2*c(evaluate(getmetric(kernel),X,C))
    end
end




#TODO CAN BE CONSIDERABLY OPTIMIZED
function update_model!(model,new_centers,μ_new,σ_new)
    if size(new_centers,1) > 0
        # model.μ = vcat(model.μ, new_vals)

        model.μ = vcat(model.μ, μ_new)
        # m_Σ = mean(diag(model.Σ))
        Σ_temp = Matrix(Diagonal(σ_new[1]*I,model.m+1))#m_Σ*Matrix{Float64}(I,model.m+size(new_centers,1),model.m+size(new_centers,1))
        Σ_temp[1:model.m,1:model.m] = model.Σ
        model.Σ = Symmetric(Σ_temp)
        # η_2temp = -0.5/m_Σ*Matrix{Float64}(I,model.m+size(new_centers,1),model.m+size(new_centers,1))
        # η_2temp[1:model.m,1:model.m] = model.η_2
        model.η_2 = -inv(model.Σ)*0.5
        # model.η_1 = vcat(model.η_1,-0.5*inv(m_Σ)*new_vals)
        model.η_1 = -0.5*model.η_2*model.μ
        model.m = length(model.μ)
        model.nFeatures = model.m
        update_matrices!(model,new_centers)
    end
end

function update_matrices!(model,new_centers)
    model.Kmm = Symmetric(kernelmatrix(model.kmeansalg.centers,model.kernel)+getvalue(model.noise)*Diagonal{Float64}(I,model.m))
    model.invKmm = inv(model.Kmm)
    model.Knm = kernelmatrix(model.X[model.MBIndices,:],model.kmeansalg.centers,model.kernel)
    model.κ = model.Knm/model.Kmm
    model.Ktilde = kerneldiagmatrix(model.X[model.MBIndices,:],model.kernel) - vec(sum(model.κ.*model.Knm,dims=2))
    model.TopMatrixForPrediction = 0
    model.DownMatrixForPrediction = 0
end

include("StreamingKMeans.jl")
include("Webscale.jl")
include("CircleKMeans.jl")
include("DataSelection.jl")
include("OfflineKMeans.jl")
####
function KLGP(mu,sig,f,sig_f)
    tot = -0.5*length(f)
    tot += 0.5*sum(log.(sig)-log.(sig_f)+(sig_f+(mu-f).^2)./sig)
    return tot
end

function JSGP(mu,sig,f,sig_f)
    tot = -0.25*length(f)
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) + (1.0 ./(sig_f)+1.0 ./(sig)).*((mu-f).^2))
end

#--------------------------------------------------------------#


#Return K inducing points from X, m being the number of Markov iterations for the seeding
function KMeansInducingPoints(X::Array{T,N},nC::Integer;nMarkov::Integer=10,kweights::Vector{T}=[0.0]) where {T,N}
    C = copy(transpose(KmeansSeed(X,nC,nMarkov)))
    if kweights!=[0.0]
        Clustering.kmeans!(copy(transpose(X)),C,weights=kweights,tol=1e-3)
    else
        Clustering.kmeans!(copy(transpose(X)),C)
    end
    return copy(transpose(C))
end

"""Fast and efficient seeding for KMeans based on [`Fast and Provably Good Seeding for k-Means](https://las.inf.ethz.ch/files/bachem16fast.pdf)"""
function KmeansSeed(X::AbstractArray{T,N},nC::Integer,nMarkov::Integer) where {T,N} #X is the data, nC the number of centers wanted, m the number of Markov iterations
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

#Compute the minimum distance
function mindistance(x::AbstractArray{T,N1},C::AbstractArray{T,N2},nC::Integer) where {T,N1,N2}#Point to look for, collection of centers, number of centers computed
  mindist = Inf
  for i in 1:nC
    mindist = min.(norm(x.-C[i])^2,mindist)
  end
  return mindist
end

# end #module
