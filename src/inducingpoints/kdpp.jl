mutable struct kDPP{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    k::Int64
    kernel::Kernel
    opt::O
    Z::M
    function kDPP(k::Int,kernel::Kernel,opt=ADAM(0.001)) where {T}
        return new{Float64,Matrix{Float64},typeof(opt)}(k,kernel,opt)
    end
end


function init!(alg::kDPP{T},X,y,kernel) where {T}
    jitt = T(Jittering())
    samp = rand(1:size(X,1))
    alg.Z = X[samp:samp,:]
    Zset = Set(samp)
    k = 1
    randorder = randperm(size(X,1))
    kᵢᵢ = kerneldiagmatrix(kernel,X,obsdim=1).+jitt
    while k < alg.k
        Xset = setdiff(1:size(X,1),Zset)
        kᵢZ = kernelmatrix(kernel,X[collect(Xset),:],alg.Z,obsdim=1)
        KZ = kernelmatrix(kernel,alg.Z,obsdim=1) + jitt*I
        Vᵢ = kᵢᵢ[collect(Xset)] - opt_diag(kᵢZ*inv(KZ),kᵢZ)
        pᵢ = Vᵢ/sum(Vᵢ)
        j = StatsBase.sample(collect(Xset),Weights(pᵢ))
        push!(Zset,j)
        alg.Z = vcat(alg.Z,X[j:j,:])
        k += 1
    end
end

function add_point!(alg::kDPP,X,y,kernel)
end

function remove_point!(alg::kDPP,K,kernel)
end
