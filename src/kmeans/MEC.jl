mutable struct MEC{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    kmax::Int64
    k::Int64
    σ::Float64
    opt::O
    invK::Matrix{Float64}
    Z::M
    function MEC(kmax::Integer=100,σ::Real=1e-3,opt=Flux.ADAM(0.001))
        return new{Float64,Matrix{Float64},typeof(opt)}(kmax,0,σ,opt)
    end
end


function init!(alg::MEC,X,y,kernel)
    @assert size(X,1) > 1 "First batch should have at least 2 samples"
    samples = sample(1:size(X,1), 2, replace=false)
    alg.Z = copy(X[samples,:])
    alg.k = size(alg.Z,1)

    while alg.k < alg.kmax
        s = sample(1:size(X,1), 59, replace=false)
        add_point!(alg, X[s,:], y[s], kernel)
    end
end

function add_point!(alg::MEC,X,y,kernel)
    b = size(X,1)
    K = kernelmatrix(alg.Z,kernel)
    Knm = kernelmatrix(X,alg.Z,kernel)
    Σ = inv(K+transpose(Knm)*Knm/(alg.σ^2)+1e-3I)
    mu = inv(alg.σ^2)*Σ*transpose(Knm)*y
    (max_err,add_i) = findmax(abs2.(Knm*mu-y))
    alg.Z = vcat(alg.Z,X[add_i,:]')
    alg.k += 1
end

function remove_point!(alg::MEC,K,kernel)

end
