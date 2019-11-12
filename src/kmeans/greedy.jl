mutable struct Greedy <: ZAlg
    k::Int64
    minibatch::Int64
    kernel::Kernel
    centers::Array{Float64,2}
    function Greedy(nInducingPoints::Integer,nMinibatch)
        return new(nInducingPoints,minibatch)
    end
end

function init!(alg::Greedy,X,y,kernel)
    @assert size(X,1)>=alg.k "Input data not big enough given $k"
    alg.centers = greedy_iterations(X,y,kernel,alg.k,alg.minibatch)
    alg.kernel = kernel
end

function greedy_iterations(X,y,kernel,k,minibatch)
    minibatch = min(size(X,1),minibatch)
    centers = zeros(0,size(X,2))
    Xset = Set(1:size(X,1))
    set_point = Set{Int64}()
    i = rand(1:size(X,1))
    centers = vcat(centers,X[i:i,:]); push!(set_point,i)
    for v in 2:k
        i = 0
        best_L = -Inf
        d = sample(collect(setdiff(Xset,set_point)),minibatch,replace=false)
        for j in d
            new_centers = vcat(centers,X[j:j,:]);
            L = ELBO_reg(new_centers,X,y,kernel)
            if L > best_L
                i = j
            end
        end
        @info "Found best L :$best_L $v/$k"
        centers = vcat(centers,X[i:i,:]); push!(set_point,i)
    end
    return centers
end

function ELBO_reg(centers,X,y,kernel)
    Knm = kernelmatrix(X,centers,kernel)
    Kmm = kernelmatrix(centers,kernel)+1e-1I
    Qnn = Symmetric(Knm*inv(Kmm)*Knm')
    Kt = kerneldiagmatrix(X,kernel) .+ 1e-1 - diag(Qnn)
    noise = 0.1
    return Distributions.logpdf(MvNormal(Matrix(Qnn+noise*I)),y)-1.0/(2*noise^2)*sum(Kt)
end

function add_point!(alg::Greedy,X,y,model)
end

function remove_point!(alg::Greedy,X,y,model)
end
