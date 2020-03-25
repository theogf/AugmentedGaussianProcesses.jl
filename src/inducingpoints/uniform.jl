mutable struct UniformSampling{T,M<:AbstractMatrix{T},O} <:
               InducingPoints{T,M,O}
    k::Int64
    opt::O
    Z::M
    function UniformSampling(nInducingPoints::Integer, opt = ADAM(0.001))
        return new{Float64,Matrix{Float64},typeof(opt)}(nInducingPoints, opt)
    end
end

function init!(alg::UniformSampling,X,y,kernel)
    @assert size(X,1)>=alg.k "Input data not big enough given $k"
    samp = sample(1:size(X,1),alg.k,replace=false)
    alg.Z = X[samp,:]
end
