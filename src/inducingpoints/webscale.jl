mutable struct Webscale{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    k::Int64
    opt::O
    v::Array{Int64,1}
    Z::M
    function Webscale(k::Int,opt=ADAM(0.001))
        return new{Float64,Matrix{Float64},typeof(opt)}(k,opt)
    end
end


function init!(alg::Webscale,X,y,kernel)
    @assert size(X,1)>=alg.k "Input data not big enough given $k"
    alg.v = zeros(Int64,alg.k);
    alg.Z = X[StatsBase.sample(1:size(X,1),alg.k,replace=false),:];
end

function add_point!(alg::Webscale,X,y,model)
    b = size(X,1)
    d = zeros(Int64,b)
    for i in 1:b
        d[i] = find_nearest_center(X[i,:],alg.Z)[1]
    end
    for i in 1:b
        alg.v[d[i]] += 1
        η = 1/alg.v[d[i]]
        alg.Z[d[i],:] = (1-η)*alg.Z[d[i],:]+ η*X[i,:]
    end
end
