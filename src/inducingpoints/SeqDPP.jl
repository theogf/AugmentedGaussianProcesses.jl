mutable struct SeqDPP{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    kernel::Kernel
    opt::O
    k::Int64
    dpp::DPP
    K::Symmetric{Float64,Matrix{Float64}}
    Z::M
    function SeqDPP(kernel::Kernel,opt=Flux.ADAM(0.001)) where {T}
        return new{Float64,Matrix{Float64},typeof(opt)}(kernel,opt)
    end
end


function init!(alg::SeqDPP{T},X,y,kernel;opt=Flux.ADAM(0.001)) where {T}
    @assert size(X,1) >= 3 "First batch should contain at least 2 elements"
    jitt = T(Jittering())
    alg.K = Symmetric(kernelmatrix(alg.kernel,X,obsdim=1)+jitt*I)
    alg.dpp = DPP(alg.K)
    samp = []
    while length(samp) < 3
        samp = rand(alg.dpp,1)[1]
    end
    alg.Z = X[samp,:]
    alg.k = length(samp)
    alg.K = Symmetric(kernelmatrix(alg.kernel,reshape(X[samp,:],alg.k,size(X,2)),obsdim=1)+jitt*I)
end
function add_point!(alg::SeqDPP{T},X,y,kernel) where {T}
    jitt = T(Jittering())
    L = Symmetric(kernelmatrix(kernel,vcat(alg.Z,X),obsdim=1)+jitt*I)
    Iₐ = diagm(vcat(zeros(alg.k),ones(size(X,1))))
    Lₐ = inv(view(inv(L+Iₐ),(alg.k+1):size(L,1),(alg.k+1):size(L,1)))-I
    new_dpp = DPP(Symmetric(Lₐ))
    new_samp = rand(new_dpp,1)[1]
    alg.Z = vcat(alg.Z,X[new_samp,:])
    alg.k += length(new_samp)
end


function add_point_old!(alg::SeqDPP,X,y,kernel)
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
