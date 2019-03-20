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
    update!(alg,X,y,model)
end

function update!(alg::OfflineKmeans,X,y,model)
    results = kmeans(Matrix(X'),alg.k)
    alg.centers = copy(results.centers')
    return results
end
