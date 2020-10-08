"""
    SeqDPP()

Sequential sampling via DeterminantalPointProcesses
"""
mutable struct SeqDPP{S,TZ<:AbstractVector{S},T} <: OnIP{S,TZ}
    k::Int
    K::Symmetric{T,Matrix{T}}
    Z::TZ
end

Base.show(io::IO, Z::SeqDPP) = print(io, "Sequential DPP")

SeqDPP() = SeqDPP(0, Symmetric(Matrix{Float64}(I(0))), [])

function SeqDPP(X::AbstractVector, k::Kernel)
    size(X, 1) > 2 || error("First batch should contain at least 3 elements")
    K = Symmetric(kernelmatrix(k, X) + jitt * I)
    dpp = DPP(K)
    samp = []
    while length(samp) < 3 # Sample from a normal DPP until at least 3 elements are samples
        samp = rand(dpp, 1)[1]
    end
    Z = Vector.(X[samp])
    m = length(samp)
    K = Symmetric(kernelmatrix(k, Z) + jitt * I)
    return SeqDPP(m, K, Z)
end


function init(Z::SeqDPP, X::AbstractVector, k::Kernel)
    return SeqDPP(X, k)
end

function add_point!(Z::SeqDPP, X::AbstractVector, k::Kernel)
    L = Symmetric(kernelmatrix(k, vcat(Z, X)) + jitt * I)
    Iₐ = diagm(vcat(zeros(Z.k), ones(size(X, 1))))
    Lₐ = inv(view(inv(L + Iₐ), (Z.k + 1):size(L, 1), (Z.k+1):size(L, 1))) - I
    new_dpp = DPP(Symmetric(Lₐ))
    new_samp = rand(new_dpp, 1)[1]
    Z = vcat(Z, Vector.(X[new_samp]))
    Z.k += length(new_samp)
end


function add_point_old!(alg::SeqDPP, X, y, kernel)
    alg.K = Symmetric(kernelmatrix(alg.Z, kernel) + 1e-7I)
    for i = 1:size(X, 1)
        k = kernelmatrix(reshape(X[i, :], 1, size(X, 2)), alg.Z, kernel)
        kk = kerneldiagmatrix(reshape(X[i, :], 1, size(X, 2)), kernel)[1]
        #using (A B; C D) = (A - C invD B, invD B; 0, I)*(I, 0; C, D)
        # p = logdet(alg.K - k'*inv(kk)*k) + logdet(kk) - (logdet(alg.K - k'*inv(kk+1)*k)+logdet(kk+1))
        p =
            logdet(alg.K - k' * inv(kk) * k) + logdet(kk) -
            (logdet(alg.K - k' * inv(kk + 1) * k) + logdet(kk + 1))
        # if p > log(alg.lim)
        if p > log(rand())
            # println(exp(p))
            alg.Z = vcat(alg.Z, X[i, :]')
            alg.K = symcat(alg.K, vec(k), kk)
            alg.k = size(alg.Z, 1)
        end
    end
end
