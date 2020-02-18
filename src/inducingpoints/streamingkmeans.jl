##From paper "An Algorithm for Online K-Means Clustering" ##


mutable struct StreamOnline{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    k_target::Int64
    opt::O
    k_efficient::Int64
    k::Int64
    f::Float64
    q::Int64
    Z::M
    function StreamOnline(k_target::Int64,opt=Flux.ADAM(0.001))
        return new{Float64,Matrix{Float64},typeof(opt)}(k_target,opt)
    end
end


function init!(alg::StreamOnline,X,y,kernel)
    @assert size(X,1)>=10 "The first batch of data should be bigger than 10 samples"
    alg.k_efficient = max(1,ceil(Int64,(alg.k_target-15)/5))
    if alg.k_efficient+10 > size(X,1)
         alg.k_efficient = 0
    end
    alg.Z = X[StatsBase.sample(1:size(X,1),alg.k_efficient+10,replace=false),:]
    # alg.centers = X[1:(alg.k_efficient+10),:]
    alg.k = alg.k_efficient+10
    w=zeros(alg.k)
    for i in 1:alg.k
        w[i] = 0.5*find_nearest_center(alg.Z[i,:],alg.Z[1:alg.k.!=i,:])[2]
    end
    alg.f = sum(sort(w)[1:10]) #Take the 10 smallest values
    alg.q = 0
end

function add_point!(alg::StreamOnline,X,y,model)
    b = size(X,1)
    for i in 1:b
        val = find_nearest_center(X[i,:],alg.Z)[2]
        if val>(alg.f*rand())
            # new_centers = vcat(new_centers,X[i,:]')
            alg.Z = vcat(alg.Z,X[i,:]')
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

"Compute the distance (kernel if included) between a point and a find_nearest_center"
function distance(X,C,k=nothing)
    if isnothing(k)
        return norm(X-C,2)^2
    else
        k(X,C)
    end
end
