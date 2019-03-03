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
export KMeansAlg, StreamOnline, Webscale, CircleKMeans, DataSelection
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
        return compute(kernel,X,X)+compute(kernel,C,C)-2*compute(kernel,X,C)
    end
end




#TODO CAN BE CONSIDERABLY OPTIMIZED
function update_model!(model,new_centers,new_vals)
    if size(new_centers,1) > 0
        model.μ = vcat(model.μ, new_vals)
        model.η₁ = vcat(model.η₁, -0.5*new_vals)
        m_Σ = mean(diag(model.Σ))
        Σ_temp = 1.0*Matrix{Float64}(I,model.m+size(new_centers,1),model.m+size(new_centers,1))
        Σ_temp[1:model.m,1:model.m] = model.Σ
        model.Σ = Σ_temp
        η₂temp = -0.5/m_Σ*Matrix{Float64}(I,model.m+size(new_centers,1),model.m+size(new_centers,1))
        η₂temp[1:model.m,1:model.m] = model.η₂
        model.η₂ = η₂temp
        model.m = length(model.μ)
        model.nFeatures = model.m
        update_matrices!(model,new_centers)
    end
end

function update_matrices!(model,new_centers)
    model.Kmm = Symmetric(kernelmatrix(model.kmeansalg.centers,model.kernel)+model.noise*Diagonal{Float64}(I,model.m))
    model.invKmm = inv(model.Kmm)
    Knm = kernelmatrix(model.X[model.MBIndices,:],model.kmeansalg.centers,model.kernel)
    model.κ = Knm/model.Kmm
    model.Ktilde = diagkernelmatrix(model.X[model.MBIndices,:],model.kernel) - sum(model.κ.*Knm,2)[:]
    model.TopMatrixForPrediction = 0
    model.DownMatrixForPrediction = 0
end

mutable struct OfflineKmeans <: KMeansAlg
    kernel::Kernel
    k::Int64
    centers::Array{Float64,2}
    function OfflineKmeans()
        return new()
    end
end

function init!(alg::OfflineKmeans,X,y,model,k::Int64)
    @assert size(X,1)>=k "Input data not big enough given $k"
    alg.k = k
end

function update!(alg::OfflineKmeans,X,y,model)
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


function init!(alg::Webscale,X,y,model,k::Int64)
    @assert size(X,1)>=k "Input data not big enough given $k"
    alg.k = k;
    alg.v = zeros(Int64,k);
    alg.centers = X[sample(1:size(X,1),k),:];
end

function update!(alg::Webscale,X,y,model)
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


function init!(alg::StreamOnline,X,y,model,k::Int64)
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

function update!(alg::StreamOnline,X,y,model)
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


function init!(alg::CircleKMeans,X,y,model,k::Int64;lim=0.9)
    @assert lim < 1.0 && lim > 0 "lim should be between 0 and 1"
    alg.centers = reshape(X[1,:],1,size(X,2))
    alg.k = 1
    update!(alg,X[2:end,:],nothing,model)
end

function update!(alg::CircleKMeans,X,y,model)
    b = size(X,1)
    for i in 1:b
        d = find_nearest_center(X[i,:],alg.centers,model.kernel)[2]
        if d>2*(1-alg.lim)
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.k += 1
        end
    end
end

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
function RandomAccept_Mean(alg,model,mu,sig,X,y)
    diff_f = 1-exp(-0.5*(mu-y)[1]^2/sig[1])
    d = find_nearest_center(X,alg.centers,model.kernel)[2]
    if d>2*(1-alg.lim[1])
        println("Distance point")
        return true
    elseif diff_f > alg.lim[2] && d<2*(1-alg.lim[1]-0.05)
        println("Likelihood point")
        return true
    end
    return false
    # println(sig[1])
    #return sig[1]>0.8*(1-diff_f)
    # return KLGP(mu[1],sig[1],y,0.001)>10
    # return (d>(1-2*alg.lim[1]) || diff_f>0.5)
    # return JSGP(mu[1],sig[1],y,0.001)>10
    # return sig[1]>0.8
    # return (0.5*sqrt(sig[1])+0.5*diff_f)>rand()
    # return 1.0*sqrt(sig[1])>rand()
end

mutable struct DataSelection <: KMeansAlg
    accepting_function::Function ###From given parameters return if point is accepted (return true or false)
    lim
    k::Int64
    centers::Array{Float64,2}
    function DataSelection(;func=RandomAccept_Mean,lim=[0.9,0.4])
        return new(func,lim)
    end
end


function init!(alg::DataSelection,X,y,model,k::Int64)
    n = size(X,1)
    init_k = max(1,ceil(Int64,n/10))
    alg.centers = reshape(X[sample(1:n,init_k,replace=false),:],init_k,size(X,2))
    alg.k = init_k
end

function update!(alg::DataSelection,X,y,model)
    b = size(X,1)
    for i in 1:b
        mu,sig = model.fstar(reshape(X[i,:],1,size(X,2)))
        # println("σ: $sig,\tabs(mu-f): $(1-exp(-0.5*(mu-y[i])[1]^2/sig[1])),\tquant: $(log(sqrt(sig))+0.5*(y[i]-mu[1])^2/sig[1])")
        if alg.accepting_function(alg,model,mu,sig,X[i,:],y[i])
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.k += 1
            update_model!(model,reshape(X[i,:],1,size(X,2)),[y[i]])
        end
    end
end

#--------------------------------------------------------------#


#Return K inducing points from X, m being the number of Markov iterations for the seeding
function KMeansInducingPoints(X::Array{T,N},nC::Integer;nMarkov::Integer=10,kweights::Vector{<:Real}=[0.0]) where {T<:Real,N}
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
