"""
    OIPS(ρ_accept=0.8, opt= ADAM(0.001); η = 0.95, kmax = Inf, ρ_remove = Inf )
    OIPS(kmax, η, opt= ADAM(0.001))

Online Inducing Points Selection.
Method from the paper include reference here.
"""
mutable struct OIPS{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    ρ_accept::Float64
    ρ_remove::Float64
    opt::O
    kmax::Float64
    η::Float64
    k::Int64
    Z::M
    function OIPS(
        ρ_accept::Real = 0.8,
        opt = ADAM(0.001);
        η::Real = 0.95,
        kmax = Inf,
        ρ_remove::Real = Inf,
    )
        @assert 0.0 <= ρ_accept <= 1.0 "ρ_accept should be between 0 and 1"
        @assert 0.0 <= η <= 1.0 "η should be between 0 and 1"
        ρ_remove = isinf(ρ_remove) ? sqrt(ρ_accept) : ρ_remove
        @assert 0.0 <= ρ_remove <= 1.0 "ρ_remove should be between 0 and 1"
        return new{Float64,Matrix{Float64},typeof(opt)}(
            ρ_accept,
            ρ_remove,
            opt,
            kmax,
            η,
        )
    end
    function OIPS(kmax::Int, η::Real = 0.98, opt = ADAM(0.001))
        @assert kmax > 0 "kmax should be bigger than 0"
        @assert 0.0 <= η <= 1.0 "η should be between 0 and 1"
        return new{Float64,Matrix{Float64},typeof(opt)}(
            0.95,
            sqrt(0.95),
            opt,
            kmax,
            η,
        )
    end
end

Base.show(io::IO, alg::OIPS) = print(
    io,
    "Online Inducing Point Selection (ρ_in : $(alg.ρ_accept), ρ_out : $(alg.ρ_remove), kmax : $(alg.kmax)).",
)

function init!(alg::OIPS, X, y, kernel)
    @assert size(X, 1) > 9 "First batch should have at least 10 samples"
    samples = sample(1:size(X, 1), 10, replace = false)
    alg.Z = copy(X[samples, :])
    alg.k = size(alg.Z, 1)
    add_point!(alg, X, y, kernel)
end

function add_point!(alg::OIPS, X, y, kernel)
    b = size(X, 1)
    for i = 1:b # Parse all points from X
        k = kernelmatrix(kernel, X[i:i, :], alg.Z, obsdim = 1)
        # d = find_nearest_center(X[i,:],alg.centers,kernel)[2]
        if maximum(k) < alg.ρ_accept #If biggest correlation is smaller than threshold add point
            alg.Z = vcat(alg.Z, X[i:i, :])
            alg.k += 1
        end
        while alg.k > alg.kmax ## If maximum number of points is reached, readapt the threshold
            K = kernelmatrix(kernel, alg.Z, obsdim = 1)
            m = maximum(K - Diagonal(K))
            @info (alg.k, alg.kmax, m)
            alg.ρ_remove = alg.η * m
            remove_point!(alg, K, kernel)
            if alg.ρ_remove < alg.ρ_accept
                alg.ρ_accept = alg.η * alg.ρ_remove
            end
            @info "ρ_accept reset to : $(alg.ρ_accept)"
        end
    end
end

function remove_point!(alg::OIPS, K, kernel)
    if alg.k > 10
        overlapcount = (x -> count(x .> alg.ρ_remove)).(eachrow(K))
        removable = SortedSet(findall(x -> x > 1, overlapcount))
        toremove = []
        c = 0
        while !isempty(removable) && alg.k > 10
            i = sample(
                collect(removable),
                Weights(overlapcount[collect(removable)]),
            )
            connected = findall(x -> x > alg.ρ_remove, K[i, :])
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
            alg.k -= 1
        end
        alg.Z = alg.Z[setdiff(1:alg.k, toremove), :]
        alg.k = size(alg.Z, 1)
    end
end
