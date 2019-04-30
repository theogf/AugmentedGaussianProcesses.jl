

mutable struct CircleKMeans <: ZAlg
    lim::Float64
    ρ::Float64 # Radius of the balls
    k::Int64
    centers::Array{Float64,2}
    function CircleKMeans(lim::Real=0.9)
        return new(lim,2*(1-lim))
    end
end


function init!(alg::CircleKMeans,X,y,kernel)
    @assert alg.lim < 1.0 && alg.lim > 0 "lim should be between 0 and 1"
    @assert size(X,1) > 1 "First batch should have at least 2 samples"
    dpp = DeterminantalPointProcess(Symmetric(kernelmatrix(X,kernel)+1e-5I))
    samples = rand(dpp,1)[1]
    if length(samples) == 1
        samples = sample(1:size(X,1),2,replace=false)
    end
    alg.centers = copy(X[samples,:])
    alg.k = size(alg.centers,1)
end

function update!(alg::CircleKMeans,X,y,kernel)
    b = size(X,1)
    for i in 1:b
        k = kernelmatrix(X[i:i,:],alg.centers,kernel)
        # d = find_nearest_center(X[i,:],alg.centers,kernel)[2]
        if maximum(k)<alg.lim*getvariance(kernel)
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.k += 1
        end
    end
end

function remove!(alg::CircleKMeans,Kmm,kernel)
    # overlaps = findall((x->count(x.>alg.lim*getvariance(kernel))).(eachcol(Kmm)).>1)
    # lowerKmm = Kmm - UpperTriangular(Kmm)
    if alg.k > 10
        ρ = alg.lim*getvariance(kernel)
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
