"""
    OIPS(ρ_accept=0.8, opt= ADAM(0.001); η = 0.95, kmax = Inf, kmin = 10, ρ_remove = Inf )
    OIPS(kmax, η = 0.98, kmin = 10)

Online Inducing Points Selection.
Method from the paper include reference here.
"""
mutable struct OIPS{S,TZ<:AbstractVector{S}} <: OnIP{S,TZ}
    ρ_accept::Float64
    ρ_remove::Float64
    kmax::Float64
    kmin::Float64
    η::Float64
    k::Int
    Z::TZ
end

Base.show(io::IO, Z::OIPS) = print(
    io,
    "Online Inducing Point Selection (ρ_in : $(Z.ρ_accept), ρ_out : $(Z.ρ_remove), kmax : $(Z.kmax))",
)

function OIPS(
    ρ_accept::Real = 0.8;
    η::Real = 0.95,
    kmax::Real = Inf,
    ρ_remove::Real = Inf,
    kmin::Real = 10.0,
)
    0.0 <= ρ_accept <= 1.0 || error("ρ_accept should be between 0 and 1")
    0.0 <= η <= 1.0 || error("η should be between 0 and 1")
    ρ_remove = isinf(ρ_remove) ? sqrt(ρ_accept) : ρ_remove
    0.0 <= ρ_remove <= 1.0 || error("ρ_remove should be between 0 and 1")
    return OIPS(
        ρ_accept,
        ρ_remove,
        kmax,
        kmin,
        η,
        0,
        [],
    )
end

function OIPS(kmax::Int, η::Real = 0.98, kmin::Real = 10)
    kmax > 0 || error("kmax should be bigger than 0")
    0.0 <= η <= 1.0 || error("η should be between 0 and 1")
    return OIPS(
        0.95,
        sqrt(0.95),
        kmax,
        kmin,
        η,
        0,
        [],
    )
end
function OIPS(Z::OIPS, X::AbstractVector)
    N = size(X, 1)
    N >= Z.kmin || error("First batch should have at least $(Z.kmin) samples")
    samples = sample(1:N, floor(Int, Z.kmin), replace = false)
    return OIPS(Z.ρ_accept, Z.ρ_remove, Z.kmax, Z.kmin, Z.η, 10, Vector.(X[samples]))
end

function init(Z::OIPS, X::AbstractVector, k::Kernel)
    Z = OIPS(Z, X)
    update!(Z, X, k)
    return Z
end

function update!(Z::OIPS, X::AbstractVector, k::Kernel)
    add_point!(Z, X, k)
end

function add_point!(Z::OIPS, X::AbstractVector, k::Kernel)
    b = size(X, 1)
    for i = 1:b # Parse all points from X
        kx = kernelmatrix(k, [X[i]], Z)
        # d = find_nearest_center(X[i,:],Z.centers,kernel)[2]
        if maximum(kx) < Z.ρ_accept #If biggest correlation is smaller than threshold add point
            Z.Z = push!(Z.Z, Vector(X[i]))
            Z.k += 1
        end
        while Z.k > Z.kmax ## If maximum number of points is reached, readapt the threshold
            K = kernelmatrix(k, Z)
            m = maximum(K - Diagonal(K))
            Z.ρ_remove = Z.η * m
            remove_point!(Z, K, k)
            if Z.ρ_remove < Z.ρ_accept
                Z.ρ_accept = Z.η * Z.ρ_remove
            end
            @info "ρ_accept reset to : $(Z.ρ_accept)"
        end
    end
end

function remove_point!(Z::OIPS, K::AbstractMatrix, kernel::Kernel)
    if Z.k > Z.kmin
        overlapcount = (x -> count(x .> Z.ρ_remove)).(eachrow(K))
        removable = SortedSet(findall(x -> x > 1, overlapcount))
        toremove = []
        c = 0
        while !isempty(removable) && Z.k > 10
            i = sample(
                collect(removable),
                Weights(overlapcount[collect(removable)]),
            )
            connected = findall(x -> x > Z.ρ_remove, K[i, :])
            overlapcount[connected] .-= 1
            outofloop = filter(x -> overlapcount[x] <= 1, connected)
            for j in outofloop
                if issubset(j, removable)
                    delete!(removable, j)
                end
            end
            push!(toremove, i)
            if issubset(i, removable)
                delete!(removable, i)
            end
            Z.k -= 1
        end
        Z.Z = Z.Z[setdiff(1:Z.k, toremove)]
        Z.k = size(Z.Z, 1)
    end
end
