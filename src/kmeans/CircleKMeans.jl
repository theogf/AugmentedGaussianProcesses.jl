

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
    alg.centers = reshape(X[1,:],1,size(X,2))
    alg.k = 1
end

function update!(alg::CircleKMeans,X,y,kernel)
    b = size(X,1)
    for i in 1:b
        k = kernelmatrix(X[i,:],alg.centers,kernel)
        # d = find_nearest_center(X[i,:],alg.centers,kernel)[2]
        if maximum(k)<alg.lim
            alg.centers = vcat(alg.centers,X[i,:]')
            alg.k += 1
        end
    end
end
