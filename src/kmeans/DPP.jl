using DeterminantalPointProcesses

mutable struct DPPAlg <: ZAlg
    lim::Float64
    kernel::Kernel
    k::Int64
    dpp::DPP
    K::Symmetric{Float64,Matrix{Float64}}
    logpdf::Float64
    centers::Array{Float64,2}
    indices
    function DPPAlg(lim,kernel)
        return new(lim,kernel)
    end
end


function init!(alg::DPPAlg,X,y,kernel)
    @assert alg.lim < 1.0 && alg.lim > 0 "lim should be between 0 and 1"
    alg.K = Symmetric(kernelmatrix(X,alg.kernel)+1e-7I)
    alg.dpp = DPP(Symmetric(kernelmatrix(X,alg.kernel)+1e-7I))
    samp = rand(alg.dpp,1)[1]
    alg.indices = [rand(1:alg.dpp.size)]
    alg.centers = X[alg.indices,:]
    # alg.centers = copy(X[samp,:])
    # alg.k = length(samp)
    alg.k = length(alg.indices)
    # alg.dpp = DPP(Symmetric(kernelmatrix(alg.centers),alg.centers)+1e-7I)
    alg.logpdf = logpmf(alg.dpp,collect(1:alg.k))
end

function update!(alg::DPPAlg,X,y,kernel)
    K = kernelmatrix(alg.centers,kernel)+1e-7I
    for i in 1:size(X,1)
        if !issubset(i,alg.indices)
            p = logdet(alg.K[vcat(alg.indices,i),vcat(alg.indices,i)]) - logdet(alg.K[vcat(alg.indices,i),vcat(alg.indices,i)]+Diagonal(vcat(falses(length(alg.indices)),true)))
            # if p > log(alg.lim)
            if p > log(rand())
                println(exp(p))
                push!(alg.indices,i)
                alg.centers = vcat(alg.centers,X[i,:]')
            end
        end
    end
    # alg.centers = X[alg.indices,:]
    alg.k = length(alg.indices)
end
