mutable struct Greedy{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    minibatch::Int64
    k::Int64
    opt::O
    Z::M
    function Greedy(nInducingPoints::Int,nMinibatch::Int,opt=Flux.ADAM(0.001))
        @assert nInducingPoints > 0
        @assert nMinibatch > 0
        return new{Float64,Matrix{Float64},typeof(opt)}(nMinibatch,nInducingPoints,opt)
    end
end

function init!(alg::Greedy,X,y,kernel)
    @assert size(X,1)>=alg.k "Input data not big enough given $k"
    alg.Z = greedy_iterations(X,y,kernel,alg.k,alg.minibatch)
end

function greedy_iterations(X,y,kernel,k,minibatch)
    minibatch = min(size(X,1),minibatch)
    Z = zeros(0,size(X,2))
    set_point = Set{Int64}()
    i = rand(1:size(X,1))
    Z = vcat(Z,X[i:i,:]); push!(set_point,i)
    for v in 2:k
        X_sub = StatsBase.sample(1:size(X,1),min(10000,size(X,1)),replace=false)
        Xset = Set(X_sub)
        i = 0
        best_L = -Inf
        d = StatsBase.sample(collect(setdiff(Xset,set_point)),minibatch,replace=false)
        for j in d
            new_Z = vcat(Z,X[j:j,:]);
            L = ELBO_reg(new_Z,X[X_sub,:],y[X_sub],kernel)
            if L > best_L
                i = j
                best_L = L
            end
        end
        @info "Found best L :$best_L $v/$k"
        Z = vcat(Z,X[i:i,:]); push!(set_point,i)
    end
    return Z
end

function ELBO_reg(Z,X,y,kernel)
    jitter = Float64(Jittering())
    Knm = kernelmatrix(kernel,X,Z,obsdim=1)
    Kmm = kernelmatrix(kernel,Z,obsdim=1)+jitter*I
    Qnn = Symmetric(Knm*inv(Kmm)*Knm')
    Kt = kerneldiagmatrix(kernel,X,obsdim=1) .+ jitter - diag(Qnn)
    noise = 0.01
    return Distributions.logpdf(MvNormal(Matrix(Qnn+noise*I)),y)-1.0/(2*noise^2)*sum(Kt)
end

function add_point!(alg::Greedy,X,y,model)
end

function remove_point!(alg::Greedy,X,y,model)
end
