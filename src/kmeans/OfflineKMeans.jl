mutable struct OfflineKmeans <: ZAlg
    k::Int64
    kernel::Kernel
    centers::Array{Float64,2}
    function OfflineKmeans(nInducingPoints::Integer)
        return new(nInducingPoints)
    end
end

function init!(alg::OfflineKmeans,X,y,kernel)
    @assert size(X,1)>=alg.k "Input data not big enough given $k"
    alg.centers = KMeansInducingPoints(X,alg.k)
end

function add_point!(alg::OfflineKmeans,X,y,model)
end

function remove_point!(alg::OfflineKmeans,X,y,model)
end
