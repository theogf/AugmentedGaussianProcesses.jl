

mutable struct CircleKMeans <: KMeansAlg
    lim::Float64
    k::Int64
    centers::Array{Float64,2}
    function CircleKMeans(;lim=0.9)
        return new(lim)
    end
end


function init!(alg::CircleKMeans,X,y,model,k::Int64;lim=0.9)
    @assert lim < 1.0 && lim > 0 "lim should be between 0 and 1"
    alg.centers = reshape(X[1,:],1,size(X,2))
    alg.k = 1
    # update!(alg,X[2:end,:],nothing,model)
    update!(alg,X,nothing,model)
end

function update!(alg::CircleKMeans,X,y,model)
    b = size(X,1)
    for i in 1:b
        d = find_nearest_center(X[i,:],alg.centers,model.kernel)[2]
        if d>2*(1-alg.lim)
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.k += 1
        end
    end
end
