"""
    GreedyIP(X::AbstractVector, m::Int, y, s, kernel, σ²)
    GreedyIP(X::AbstractMatrix, m::Int, y, s, kernel, σ²; obsdim = 1)

 - `X` is the input data
 - `m` is the desired number of inducing points
 - `y` is the output data
 - `s` is the minibatch size on which to select a new inducing point
 - `σ²` is the likelihood noise

Greedy approach first proposed by Titsias[1].
Algorithm loops over minibatches of data and select the best ELBO improvement.

[1] Titsias, M. Variational Learning of Inducing Variables in Sparse Gaussian Processes. Aistats 5, 567–574 (2009).
"""
mutable struct GreedyIP{S,TZ<:AbstractVector{S}} <: OffIP{S,TZ}
    s::Int
    k::Int
    Z::TZ
end

function GreedyIP(
    X::AbstractMatrix,
    m::Int,
    y::AbstractVector,
    s::Int,
    kernel::Kernel,
    σ²::Real;
    obsdim::Int = 1,
    )
    GreedyIP(
        KernelFunctions.vec_of_vecs(X, obsdim=obsdim),
        m::Int,
        y::AbstractVector,
        s::Int,
        kernel::Kernel,
        σ²::Real,
    )
end

function GreedyIP(
    X::AbstractVector,
    m::Int,
    y::AbstractVector,
    S::Int,
    kernel::Kernel,
    σ²::Real,
)
    m > 0 || error("Number of inducing points should be positive")
    S > 0 || error("Size of the minibatch should be positive")
    σ² > 0 || error("Noise should be positive")
    Z = greedy_ip(X, y, kernel, m, S, σ², obsdim)
    return GreedyIP(
        S,
        m,
        Z
    )
end

Base.show(io::IO, alg::GreedyIP) =
    print(io, "Greedy Selection of Inducing Points")

function greedy_ip(X::AbstractVector, y::AbstractVector, kernel::Kernel, m, S, σ², )
    T = eltype(X)
    N = size(X, 1)
    Z = Vector{eltype(X)}() #Initialize array of IPs
    IP_set = Set{Int}() # Keep track of selected points
    i = rand(1:N) # Take a random initial point
    push!(Z, Vector(X[i])); push!(IP_set, i)
    for v = 2:m
        # Evaluate on a subset of the points of a maximum size of 1000
        X_te = sample(1:N, min(1000, N), replace = false)
        X_te_set = Set(X_te)
        i = 0
        best_L = -Inf
        # Parse over random points of this subset
        new_candidates = collect(setdiff(X_te_set, IP_set))
        d = sample(
            collect(setdiff(X_te_set, IP_set)),
            min(S, length(new_candidates)),
            replace = false,
        )
        for j in d
            new_Z = vcat(Z, X[j])
            L = elbo(new_Z, X[X_sub], y[X_sub], kernel, σ²)
            if L > best_L
                i = j
                best_L = L
            end
        end
        @info "Found best L :$best_L $v/$m"
        push!(Z, Vector(X[i]))
        push!(IP_set, i)
    end
    return Z
end

function elbo(Z::AbstractVector, X::AbstractVector, y::AbstractVector, kernel::Kernel, σ²::Real)
    Knm = kernelmatrix(kernel, X, Z)
    Kmm = kernelmatrix(kernel, Z) + T(jitt) * I
    Qff = Knm * (Kmm \ Knm')
    Kt = kerneldiagmatrix(kernel, X) .+ T(jitt) - diag(Qff)
    Σ = inv(Kmm) + Knm' * Knm / σ²
    invQnn = 1/σ² * I - 1/ (σ²)^2 * Knm * inv(Σ) * Knm'
    logdetQnn = logdet(Σ) + logdet(Kmm)
    return -0.5 * dot(y, invQnn * y) - 0.5 * logdetQnn -
           1.0 / (2 * σ²) * sum(Kt)
end
