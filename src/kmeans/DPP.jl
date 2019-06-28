using DeterminantalPointProcesses

mutable struct DPPAlg <: ZAlg
    lim::Float64
    kernel::Kernel
    k::Int64
    dpp::DPP
    K::Symmetric{Float64,Matrix{Float64}}
    centers::Array{Float64,2}
    optimizers::Vector{Optimizer}
    function DPPAlg(lim,kernel)
        return new(lim,kernel)
    end
end


function init!(alg::DPPAlg,X,y,kernel;optimizer=Adam())
    @assert alg.lim < 1.0 && alg.lim > 0 "lim should be between 0 and 1"
    @assert size(X,1) >= 1 "First batch should contain at least 2 elements"
    alg.K = Symmetric(kernelmatrix(X,alg.kernel)+1e-7I)
    alg.dpp = DPP(Symmetric(kernelmatrix(X,alg.kernel)+1e-7I))
    samp = rand(alg.dpp,1,2)[1]
    alg.centers = X[samp,:]
    # alg.centers = copy(X[samp,:])
    # alg.k = length(samp)
    alg.k = length(samp)
    alg.optimizers = [deepcopy(optimizer) for _ in 1:alg.k]

    # alg.dpp = DPP(Symmetric(kernelmatrix(alg.centers),alg.centers)+1e-7I)
    alg.K = Symmetric(kernelmatrix(reshape(X[samp,:],alg.k,size(X,2)),alg.kernel)+1e-7I)
end

function add_point!(alg::DPPAlg,X,y,kernel;optimizer=Adam())
    alg.K = Symmetric(kernelmatrix(alg.centers,kernel)+1e-7I)
    for i in 1:size(X,1)
        k = kernelmatrix(reshape(X[i,:],1,size(X,2)),alg.centers,kernel)
        kk = kerneldiagmatrix(reshape(X[i,:],1,size(X,2)),kernel)[1]
        #using (A B; C D) = (A - C invD B, invD B; 0, I)*(I, 0; C, D)
        # p = logdet(alg.K - k'*inv(kk)*k) + logdet(kk) - (logdet(alg.K - k'*inv(kk+1)*k)+logdet(kk+1))
        p = logdet(alg.K - k'*inv(kk)*k) + logdet(kk) - (logdet(alg.K - k'*inv(kk+1)*k)+logdet(kk+1))
        # if p > log(alg.lim)
        if p > log(rand())
            # println(exp(p))
            push!(alg.optimizers,deepcopy(optimizer))
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.K = symcat(alg.K,vec(k),kk)
            alg.k = size(alg.centers,1)
        end
    end
end

function remove_point!(alg::DPPAlg,Kmm,kernel)
    # # overlaps = findall((x->count(x.>alg.lim*getvariance(kernel))).(eachcol(Kmm)).>1)
    # # lowerKmm = Kmm - UpperTriangular(Kmm)
    # ρ = alg.lim*getvariance(kernel)
    # overlapcount = (x->count(x.>ρ)).(eachrow(Kmm))
    # removable = SortedSet(findall(x->x>1,overlapcount))
    # toremove = []
    # c = 0
    # while !isempty(removable)
    #     i = sample(collect(removable),Weights(overlapcount[collect(removable)]))
    #     connected = findall(x->x>ρ,Kmm[i,:])
    #     overlapcount[connected] .-= 1
    #     outofloop = filter(x->overlapcount[x]<=1,connected)
    #     for j in outofloop
    #         if issubset(j,removable)
    #             delete!(removable,j)
    #         end
    #     end
    #     push!(toremove,i)
    #     if issubset(i,removable)
    #         delete!(removable,i)
    #     end
    # end
    # alg.centers = alg.centers[setdiff(1:alg.k,toremove),:]
    # alg.k = size(alg.centers,1)
end
