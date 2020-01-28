mutable struct StdDPP{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    kernel::Kernel
    opt::O
    k::Int64
    Z::M
    function StdDPP(kernel::Kernel,opt=Flux.ADAM(0.001)) where {T}
        return new{Float64,Matrix{Float64},typeof(opt)}(kernel,opt)
    end
end


function init!(alg::StdDPP{T},X,y,kernel) where {T}
    jitt = T(Jittering())
    K = Symmetric(kernelmatrix(alg.kernel,X,obsdim=1)+jitt*I)
    dpp = DPP(K)
    samp = rand(dpp,1)[1]
    alg.Z = X[samp,:]
    alg.k = length(samp)
end

function add_point!(alg::StdDPP,X,y,kernel)
end

function remove_point!(alg::StdDPP,K,kernel)
end
