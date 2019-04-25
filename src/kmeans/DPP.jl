using DeterminantalPointProcesses

mutable struct DPPAlg <: ZAlg
    lim::Float64
    kernel::Kernel
    k::Int64
    dpp::DPP
    K::Symmetric{Float64,Matrix{Float64}}
    centers::Array{Float64,2}
    function DPPAlg(lim,kernel)
        return new(lim,kernel)
    end
end


function init!(alg::DPPAlg,X,y,kernel)
    @assert alg.lim < 1.0 && alg.lim > 0 "lim should be between 0 and 1"
    alg.K = Symmetric(kernelmatrix(X,alg.kernel)+1e-7I)
    alg.dpp = DPP(Symmetric(kernelmatrix(X,alg.kernel)+1e-7I))
    samp = rand(alg.dpp,1)[1]
    alg.centers = X[samp,:]
    # alg.centers = copy(X[samp,:])
    # alg.k = length(samp)
    alg.k = length(samp)
    # alg.dpp = DPP(Symmetric(kernelmatrix(alg.centers),alg.centers)+1e-7I)
    alg.K = Symmetric(kernelmatrix(reshape(X[samp,:],alg.k,size(X,2)),alg.kernel)+1e-7I)
end

function update!(alg::DPPAlg,X,y,kernel)
    for i in 1:size(X,1)
        k = kernelmatrix(reshape(X[i,:],1,size(X,2)),alg.centers,kernel)
        kk = kerneldiagmatrix(reshape(X[i,:],1,size(X,2)),kernel)[1]
        #using (A B; C D) = (A - C invD B, invDB; 0, I)*(I, 0; C, D)
        p = logdet(alg.K - k'*inv(kk)*k) + logdet(kk) - (logdet(alg.K - k'*inv(kk+1)*k)+logdet(kk+1))
        # if p > log(alg.lim)
        if p > log(rand())
            # println(exp(p))
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.K = symcat(alg.K,vec(k),kk)
            alg.k = size(alg.centers,1)
        end
    end
end
