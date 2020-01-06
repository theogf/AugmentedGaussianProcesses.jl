using DeterminantalPointProcesses

mutable struct DPPAlg{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    lim::Float64
    kernel::Kernel
    opt::O
    k::Int64
    dpp::DPP
    K::Symmetric{Float64,Matrix{Float64}}
    Z::M
    function DPPAlg(lim,kernel::Kernel{T},opt=Flux.ADAM(0.001)) where {T}
        return new{T,Matrix{T},typeof(opt)}(lim,kernel,opt)
    end
end


function init!(alg::DPPAlg,X,y,kernel;opt=Flux.ADAM(0.001))
    @assert alg.lim < 1.0 && alg.lim > 0 "lim should be between 0 and 1"
    @assert size(X,1) >= 1 "First batch should contain at least 2 elements"
    alg.K = Symmetric(kernelmatrix(X,alg.kernel)+1e-7I)
    alg.dpp = DPP(Symmetric(kernelmatrix(X,alg.kernel)+1e-7I))
    samp = rand(alg.dpp,1,2)[1]
    alg.Z = X[samp,:]
    # alg.Z = copy(X[samp,:])
    # alg.k = length(samp)
    alg.k = length(samp)
    # alg.dpp = DPP(Symmetric(kernelmatrix(alg.Z),alg.Z)+1e-7I)
    alg.K = Symmetric(kernelmatrix(reshape(X[samp,:],alg.k,size(X,2)),alg.kernel)+1e-7I)
end

function add_point!(alg::DPPAlg,X,y,kernel)
    alg.K = Symmetric(kernelmatrix(alg.Z,kernel)+1e-7I)
    for i in 1:size(X,1)
        k = kernelmatrix(reshape(X[i,:],1,size(X,2)),alg.Z,kernel)
        kk = kerneldiagmatrix(reshape(X[i,:],1,size(X,2)),kernel)[1]
        #using (A B; C D) = (A - C invD B, invD B; 0, I)*(I, 0; C, D)
        # p = logdet(alg.K - k'*inv(kk)*k) + logdet(kk) - (logdet(alg.K - k'*inv(kk+1)*k)+logdet(kk+1))
        p = logdet(alg.K - k'*inv(kk)*k) + logdet(kk) - (logdet(alg.K - k'*inv(kk+1)*k)+logdet(kk+1))
        # if p > log(alg.lim)
        if p > log(rand())
            # println(exp(p))
            alg.Z = vcat(alg.Z,X[i,:]')
            alg.K = symcat(alg.K,vec(k),kk)
            alg.k = size(alg.Z,1)
        end
    end
end

function remove_point!(alg::DPPAlg,K,kernel)
end
