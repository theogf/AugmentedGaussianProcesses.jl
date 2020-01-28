using DeterminantalPointProcesses

mutable struct kDPP{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    k::Int64
    kernel::Kernel
    opt::O
    Z::M
    function kDPP(k::Int,kernel::Kernel,opt=Flux.ADAM(0.001)) where {T}
        return new{Float64,Matrix{Float64},typeof(opt)}(k,kernel,opt)
    end
end


function init!(alg::kDPP{T},X,y,kernel) where {T}
    jitt = T(Jittering())
    samp = rand(1:size(X,1))
    alg.Z = X[samp:samp,:]
    k = 1; i = 1
    randorder = randperm(size(X,1))
    tZᵢ = [1]
    while k < alg.k && i < size(X,1)
        kᵢᵢ = kernelmatrix(kernel,X[i:i,:],obsdim=1)
        kᵢZ = kernelmatrix(kernel,X[i:i,:],alg.Z,obsdim=1)
        Vᵢ = kᵢᵢ - kᵢZ*tZᵢ
        if false
            alg.Z = vcat(Z,X[i:i,:])
            κ = vcat()
            k = size(alg.Z,1)
            tZᵢ = vcat(tZᵢ+1/r*(tZᵢ*tZᵢ'*kᵢZ'-tZᵢ))
        end
        i += 1
    end
end

function add_point!(alg::kDPP,X,y,kernel)
end

function remove_point!(alg::kDPP,K,kernel)
end
