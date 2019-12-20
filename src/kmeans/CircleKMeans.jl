mutable struct CircleKMeans{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    ρ_accept::Float64
    ρ_remove::Float64
    opt::O
    k::Int64
    Z::M
    function CircleKMeans(ρ_accept::Real=0.8,ρ_remove::Real=1.0,opt=Flux.ADAM(0.001))
        @assert 0.0 <= ρ_accept <= 1.0 "ρ_accept should be between 0 and 1"
        @assert 0.0 <= ρ_remove <= 1.0 "ρ_remove should be between 0 and 1"
        return new(ρ_accept,ρ_remove,opt)
    end
end


function init!(alg::CircleKMeans,X,y,kernel)
    @assert size(X,1) > 9 "First batch should have at least 10 samples"
    samples = sample(1:size(X,1),10,replace=false)
    alg.Z = copy(X[samples,:])
    alg.k = size(alg.Z,1)
    add_point!(alg,X,y,kernel)
end

function add_point!(alg::CircleKMeans,X,y,kernel)
    b = size(X,1)
    for i in 1:b
        k = kernelmatrix(kernel,X[i:i,:],alg.Z,obsdim=1)
        # d = find_nearest_center(X[i,:],alg.centers,kernel)[2]
        if maximum(k) < alg.ρ_accept
            alg.Z = vcat(alg.Z,X[i,:]')
            alg.k += 1
        end
    end
end

function remove_point!(alg::CircleKMeans,K,kernel)
    # overlaps = findall((x->count(x.>alg.lim*getvariance(kernel))).(eachcol(Kmm)).>1)
    # lowerKmm = Kmm - UpperTriangular(Kmm)
    if alg.k > 10
        overlapcount = (x->count(x.>alg.ρ_remove)).(eachrow(K))
        removable = SortedSet(findall(x->x>1,overlapcount))
        toremove = []
        c = 0
        while !isempty(removable)
            i = sample(collect(removable),Weights(overlapcount[collect(removable)]))
            connected = findall(x->x>alg.ρ_remove,K[i,:])
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
        alg.Z = alg.Z[setdiff(1:alg.k,toremove),:]
        alg.k = size(alg.Z,1)
    end
end


function update_centers!(alg,Zgrad)
    for i in 1:alg.k
        alg.Z[i,:] .+= apply!.([alg.opt],eachrow(alg.Z),eachrow(Zgrad))
    end
end
