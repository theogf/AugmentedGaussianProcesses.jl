### DeterminantalPointProcesses.jl code snippet as package is not compatible with  Julia >= 1.0. See https://github.com/alshedivat/DeterminantalPointProcesses.jl
###

### Struct

mutable struct DeterminantalPointProcess
    L::Symmetric
    Lfact::Eigen
    size::Int
    rng::AbstractRNG

    function DeterminantalPointProcess(L::Symmetric; seed::Int=42)
        Lfact = eigen(L)
        new(L, Lfact, length(Lfact.values), MersenneTwister(seed))
    end

    function DeterminantalPointProcess(Lfact::Eigen; seed::Int=42)
        L = Symmetric((Lfact.vectors .* Lfact.values') * Lfact.vectors')
        new(L, Lfact, length(Lfact.values), MersenneTwister(seed))
    end
end

### Sampling

"""Exact sampling from a DPP [1].
"""
function Base.rand(dpp::DeterminantalPointProcess, N::Int)
    Λ = AbstractArray{Float64}(dpp.Lfact.values)
    V = AbstractMatrix{Float64}(dpp.Lfact.vectors)
    M = AbstractMatrix{Bool}(zeros(Bool, dpp.size, N))

    # step I: sample masks for elementary DPPs
    pmap((i, seed) -> _sample_mask(Λ, M, i, seed),
         1:N, abs.(rand(dpp.rng, Int, N)))

    # step II: iteratively sample from a mixture of elementary DPPs
    pmap((i, seed) -> _sample_from_elementary(V, M, i, seed),
         1:N, abs.(rand(dpp.rng, Int, N)))
end

"""Exact sampling from a k-DPP [1].
"""
function Base.rand(dpp::DeterminantalPointProcess, N::Int, k::Int)
    Λ = AbstractArray{Float64}(dpp.Lfact.values)
    V = AbstractMatrix{Float64}(dpp.Lfact.vectors)
    M = AbstractMatrix{Bool}(zeros(Bool, dpp.size, N))

    # compute elementary symmetric polynomials
    E = AbstractMatrix{Float64}(elem_symm_poly(dpp.Lfact.values, k))

    # step I: sample masks for elementary DPPs
    pmap((i, seed) -> _sample_k_mask(Λ, M, E, k, i, seed),
         1:N, abs.(rand(dpp.rng, Int, N)))

    # step II: iteratively sample from a mixture of elementary DPPs
    pmap((i, seed) -> _sample_from_elementary(V, M, i, seed),
         1:N, abs.(rand(dpp.rng, Int, N)))
end


"""Sample a mask for an elementary DPP.
"""
function _sample_mask(Λ::AbstractArray{Float64},
                      M::AbstractMatrix{Bool},
                      i::Int, seed::Int)
    rng = MersenneTwister(seed)

    for j in 1:length(Λ)
        M[j, i] = (rand(rng) < Λ[j] / (Λ[j] + 1))
    end
end

"""Sample a mask for an elementary k-DPP.
"""
function _sample_k_mask(Λ::AbstractArray{Float64},
                        M::AbstractMatrix{Bool},
                        E::AbstractMatrix{Float64},
                        k::Int, i::Int, seed::Int)
    rng = MersenneTwister(seed)

    j = length(Λ)
    remaining = k

    # iteratively sample a k-mask
    while remaining > 0
        # compute marginal of j given that we choose remaining values from 1:j
        if j == remaining
            marg = 1
        else
            marg = Λ[j] * E[remaining, j] / E[remaining + 1, j + 1];
        end

        # sample marginal
        if rand(rng) <= marg
            M[j, i] = true
            remaining -= 1
        end
        j -= 1
      end
end

"""Exact sampling from an elementary DPP. The algorithm based on [1].
"""
function _sample_from_elementary(V::AbstractMatrix,
                                 M::AbstractMatrix{Bool},
                                 i::Int, seed::Int)
    rng = MersenneTwister(seed)

    # select the elementary DPP
    V_mask = M[:, i]

    # edge case: empty sample
    if !any(V_mask)
        return Int[]
    end

    # select the kernel of the elementary DPP
    L = V[:, V_mask]

    Y = Int[]
    mask = ones(Bool, size(L, 2))
    prob = Array{Float64}(undef,size(L, 1))

    for i in 1:size(L, 2)
        # compute probabilities
        fill!(prob, 0)
        for c in 1:size(L, 2)
            !mask[c] && continue
            for r in 1:size(L, 1)
                prob[r] += L[r, c].^2
            end
        end
        prob ./= sum(prob)

        # sample a point in the original space
        h = findfirst(rand(rng) .<= cumsum(prob))
        push!(Y, h)

        # select and mask-out an element
        j = get_first_nz_idx(L[h, :], mask)
        mask[j] = false

        if any(mask)
            # Subtract scaled Lj from other columns so that their
            # projections on e_s[i] turns into 0. This operation
            # preserves the rank of L_{-j}.
            for c in 1:size(L, 2)
                !mask[c] && continue
                for r in 1:size(L, 1)
                    L[r, c] -= L[r, j] * L[h, c] / L[h, j]
                end
            end

            # Gram-Schmidt orthogonalization
            L[:, mask] = Matrix(qr(L[:, mask]).Q)
        end
    end

    sort(Y)
end


### PDF functions


"""Compute the log probability of a sample `z` under the given DPP.
"""
function logpmf(dpp::DeterminantalPointProcess, z::Array{Int})
    L_z_eigvals = eigvals(dpp.L[z, z])
    return sum(log.(L_z_eigvals)) - sum(log.(dpp.Lfact.values .+ 1))
end

"""Compute the log probability of a sample `z` under the given k-DPP.
"""
function logpmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    L_z_eigvals = eigvals(dpp.L[z, z])
    return sum(log.(L_z_eigvals)) .- log(elem_symm_poly(dpp.Lfact.values, k)[end, end])
end

"""Compute the probability of a sample `z` under the given DPP.
"""
function pmf(dpp::DeterminantalPointProcess, z::Array{Int})
    exp(logpmf(dpp, z))
end

"""Compute the probability of a sample `z` under the given DPP.
"""
function pmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    exp(logpmf(dpp, z, k))
end
