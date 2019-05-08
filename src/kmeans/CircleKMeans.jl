mutable struct CircleKMeans <: ZAlg
    ρ_accept::Float64
    ρ_remove::Float64
    k::Int64
    centers::Array{Float64,2}
    function CircleKMeans(lim::Real=0.8;lim_rem::Real=0.9)
        return new(lim,lim_rem)
    end
end


function init!(alg::CircleKMeans,X,y,kernel)
    @assert alg.ρ_accept < 1.0 && alg.ρ_accept > 0 "lim should be between 0 and 1"
    @assert alg.ρ_remove < 1.0 && alg.ρ_remove > 0 "lim should be between 0 and 1"
    @assert size(X,1) > 1 "First batch should have at least 2 samples"
    samples = sample(1:size(X,1),2,replace=false)
    alg.centers = copy(X[samples,:])
    alg.k = size(alg.centers,1)
    update!(alg,X,y,kernel)
end

function update!(alg::CircleKMeans,X,y,kernel)
    b = size(X,1)
    ρ = alg.ρ_accept*getvariance(kernel)
    for i in 1:b
        k = kernelmatrix(X[i:i,:],alg.centers,kernel)
        # d = find_nearest_center(X[i,:],alg.centers,kernel)[2]
        if maximum(k) < ρ
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.k += 1
        end
    end
end

function remove!(alg::CircleKMeans,Kmm,kernel)
    # overlaps = findall((x->count(x.>alg.lim*getvariance(kernel))).(eachcol(Kmm)).>1)
    # lowerKmm = Kmm - UpperTriangular(Kmm)
    if alg.k > 10
        ρ = alg.ρ_remove*getvariance(kernel)
        overlapcount = (x->count(x.>ρ)).(eachrow(Kmm))
        removable = SortedSet(findall(x->x>1,overlapcount))
        toremove = []
        c = 0
        while !isempty(removable)
            i = sample(collect(removable),Weights(overlapcount[collect(removable)]))
            connected = findall(x->x>ρ,Kmm[i,:])
            overlapcount[connected] .-= 1
            outofloop = filter(x->overlapcount[x]<=1,connected)
            for j in outofloop
                if issubset(j,removable)
                    delete!(removable,j)
                end
            end
            push!(toremove,i)
            if issubset(i,removable)
                delete!(removable,i)
            end
        end
        alg.centers = alg.centers[setdiff(1:alg.k,toremove),:]
        alg.k = size(alg.centers,1)
    end
end
