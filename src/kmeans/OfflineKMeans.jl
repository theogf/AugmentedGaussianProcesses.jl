mutable struct OfflineKmeans{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    k::Int64
    opt::O
    nMarkov::Int64
    Z::M
    function OfflineKmeans(nInducingPoints::Integer,opt=Flux.ADAM(0.001);nMarkov=10)
        return new{Float64,Matrix{Float64},typeof(opt)}(nInducingPoints,opt,nMarkov)
    end
end

function init!(alg::OfflineKmeans,X,y,kernel;tol=1e-3)
    @assert size(X,1)>=alg.k "Input data not big enough given $k"
    alg.Z = KMeansInducingPoints(X,alg.k,nMarkov=alg.nMarkov,tol=tol)
end

function add_point!(alg::OfflineKmeans,X,y,model)
end

function remove_point!(alg::OfflineKmeans,X,y,model)
end
