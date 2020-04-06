mutable struct Greedy{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    minibatch::Int64
    k::Int64
    opt::O
    σ²::Float64
    Z::M
    function Greedy(
        nInducingPoints::Int,
        nMinibatch::Int;
        opt = ADAM(0.001),
        σ²::Real = 0.01,
    )
        @assert nInducingPoints > 0
        @assert nMinibatch > 0
        return new{Float64,Matrix{Float64}, typeof(opt)}(
            nMinibatch,
            nInducingPoints,
            opt,
            σ²,
        )
    end
end

Base.show(io::IO, alg::Greedy) =
    print(io, "Greedy Selection of Inducing Points")

function init!(alg::Greedy, X, y, kernel)
    @assert size(X, 1) >= alg.k "Input data not big enough given $(alg.k)"
    @assert size(X, 1) >= alg.minibatch "Minibatch size too large for the dataset"
    alg.Z = greedy_iterations(X, y, kernel, alg.k, alg.minibatch, alg.σ²)
end

function greedy_iterations(X, y, kernel, k, minibatch, noise)
    Z = zeros(0, size(X, 2)) #Initialize array
    set_point = Set{Int64}() # Keep track of selected points
    i = rand(1:size(X, 1)) # Take a random initial point
    Z = vcat(Z, X[i:i, :])
    push!(set_point, i)
    for v = 2:k
        # Evaluate on a subset of the points of a maximum size of 1000
        X_sub = sample(1:size(X, 1), min(1000, size(X, 1)), replace = false)
        Xset = Set(X_sub)
        i = 0
        best_L = -Inf
        # Parse over random points of this subset
        new_candidates = collect(setdiff(Xset, set_point))
        d = sample(
            collect(setdiff(Xset, set_point)),
            min(minibatch, length(new_candidates)),
            replace = false,
        )
        for j in d
            new_Z = vcat(Z, X[j:j, :])
            L = ELBO_reg(new_Z, X[X_sub, :], y[X_sub], kernel, noise)
            if L > best_L
                i = j
                best_L = L
            end
        end
        @info "Found best L :$best_L $v/$k"
        Z = vcat(Z, X[i:i, :])
        push!(set_point, i)
    end
    return Z
end

function ELBO_reg(Z::AbstractArray{T}, X, y, kernel, noise) where {T}
    Knm = kernelmatrix(kernel, X, Z, obsdim = 1)
    Kmm = kernelmatrix(kernel, Z, obsdim = 1) + T(jitt) * I
    Qff = Symmetric(Knm * inv(Kmm) * Knm')
    Kt = kerneldiagmatrix(kernel, X, obsdim = 1) .+ T(jitt) - diag(Qff)
    Σ = inv(Kmm) + noise^(-2) * Knm' * Knm
    invQnn = noise^(-2) * I - noise^(-4) * Knm * inv(Σ) * Knm'
    logdetQnn = logdet(Σ) + logdet(Kmm)
    return -0.5 * dot(y, invQnn * y) - 0.5 * logdetQnn -
           1.0 / (2 * noise^2) * sum(Kt)
end
