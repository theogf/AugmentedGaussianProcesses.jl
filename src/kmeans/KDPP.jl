using DeterminantalPointProcesses

mutable struct kDPP{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    kernel::Kernel
    opt::O
    k::Int64
    dpp::DPP
    Z::M
    function kDPP(lim,kernel::Kernel{T},opt=Flux.ADAM(0.001)) where {T}
        return new{T,Matrix{T},typeof(opt)}(kernel,opt)
    end
end


function init!(alg::kDPP{T},X,y,kernel) where {T}
    jitt = T(Jittering())
    K = Symmetric(kernelmatrix(alg.kernel,X,obsdim=1)+jitt*I)
    alg.dpp = DPP(K)
    samp = rand(alg.dpp,1,k)
    alg.Z = X[samp,:]
    alg.k = length(samp)
end

function add_point!(alg::kDPP,X,y,kernel)
end

function remove_point!(alg::kDPP,K,kernel)
end
