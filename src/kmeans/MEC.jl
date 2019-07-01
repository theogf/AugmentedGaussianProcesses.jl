mutable struct MEC <: ZAlg
    kmax::Int64
    k::Int64
    σ::Float64
    invK::Matrix{Float64}
    centers::Array{Float64,2}
    function MEC(kmax::Integer=100,σ::Real=1e-3)
        return new(kmax,0,σ)
    end
end


function init!(alg::MEC,X,y,kernel)
    @assert size(X,1) > 1 "First batch should have at least 2 samples"
    samples = sample(1:size(X,1), 2, replace=false)
    alg.centers = copy(X[samples,:])
    alg.k = size(alg.centers,1)

    while alg.k < alg.kmax
        s = sample(1:size(X,1), 59, replace=false)
        add_point!(alg, X[s,:], y[s], kernel)
    end
end

function add_point!(alg::MEC,X,y,kernel)
    b = size(X,1)
    K = kernelmatrix(alg.centers,kernel)
    Knm = kernelmatrix(X,alg.centers,kernel)
    Σ = inv(K+transpose(Knm)*Knm/(alg.σ^2)+1e-3I)
    mu = inv(alg.σ^2)*Σ*transpose(Knm)*y
    (max_err,add_i) = findmax(abs2.(Knm*mu-y))
    alg.centers = vcat(alg.centers,X[add_i,:]')
    alg.k += 1
end

function remove_point!(alg::MEC,Kmm,kernel)

end
