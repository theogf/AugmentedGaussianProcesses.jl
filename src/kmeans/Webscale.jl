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
