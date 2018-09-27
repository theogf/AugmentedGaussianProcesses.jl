"""
    Create the kernel matrix from the training data or the correlation matrix one of set of vectors
"""
function oldkernelmatrix!(K::Matrix{T},X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T}) where {T,N}
    (n1,n2) = size(K)
    @assert n1==size(X1,1)
    @assert n2==size(X2,1)
    @inbounds for i in 1:n1, j in 1:n2
         K[i,j] = compute(kernel,X1[i,:],X2[j,:])
    end
    return K
end

function kernelmatrix(X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(metric(kernel),X1',X2')
    @inbounds for i in eachindex(K)
        K[i] = compute!(kernel,K[i])
    end
    return getvalue(kernel.variance)*K
end

function kernelmatrix!(K::Array{T,N},X1::Array{T,N},X2::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X1,1)
    @assert n2==size(X2,1)
    pairwise!(K',metric(kernel),X2',X1')
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,K[i])
    end
    return getvalue(kernel.variance)*K
end

function kernelmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    K = pairwise(metric(kernel),X')
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,K[i])
    end
    return getvalue(kernel.variance)*K
end


function kernelmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    (n1,n2) = size(K)
    @assert n1==size(X,1)
    @assert n1==n2
    pairwise!(K',metric(kernel),X')
    @inbounds for i in in eachindex(K)
        K[i] = compute(kernel,K[i])
    end
    return getvalue(kernel.variance)*K
end

function kerneldiagmatrix(X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(X,1)
    K = zeros(T,n)
    @inbounds for i in eachindex(K)
        K[i] = compute(kernel,evaluate(metric(kernel),X[i,:],X[i,:]))
    end
    return getvalue(kernel.variance)*K
end

function kerneldiagmatrix!(K::Array{T,N},X::Array{T,N},kernel::Kernel{T,KT}) where {T,N,KT}
    n = size(K,1)
    @assert n == size(X,1)
    @inbounds for i in 1:n
        K[i] = compute(kernel,evaluate(metric(kernel),X[i,:],X[i,:]))
    end
    return getvalue(kernel.variance)*K
end


function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    K = pairwise(metric(kernel),X')
    Jl = zero(K)
    l = getvalue(kernel.lengthscales[1])
    @inbounds for i in eachindex(K)
        Jl[i] = K[i]/l
        K[i] = compute(kernel,K[i])
    end
    return [getvalue(kernel.variance)*Jl.*K,K]
end

"When K has already been computed"
function kernelderivativematrix(X::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(metric(kernel),X')
    Jl = zero(K)
    l = getvalue(kernel.lengthscales[1])
    @inbounds for i in eachindex(K)
        Jl[i] = P[i]/l
    end
    return [Jl.*K,K./getvalue(kernel.variance)]
end

function kernelderivativematrix(X::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvalue(kernel.variance)
    ls = vcat(getvalue(kernel.lengthscales),v)
    K = pairwise(metric(kernel),X')
    Ki = vcat([pairwise(SqEuclidean(),X[:,i]') for i in 1:kernel.Ndim],[fill(one(T),size(K))])
    Jl = [fill(one(T),size(K)) for _ in 1:(kernel.Ndim+1)]
    @inbounds for j in eachindex(K)
        K[j] = compute(kernel,K[j])
        broadcast((jl,ki,l)->jl[j] = v*ki[j]/l*K[j],
                    Jl,Ki,ls)
    end
    # Jl .= v* Ki
    return Jl
end

function kernelderivativematrix(X::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,ARDKernel}) where {T,N}
    v = getvalue(kernel.variance)
    ls = vcat(getvalue(kernel.lengthscales),v)
    Ki = vcat([pairwise(SqEuclidean(),X[:,i]') for i in 1:kernel.Ndim],[fill(one(T),size(K))])
    Jl = [fill(one(T),size(K)) for _ in 1:(kernel.Ndim+1)]
    @inbounds for j in eachindex(K)
        broadcast((jl,ki,l)->jl[j] = v*ki[j]/l*K[j],Jl,Ki,ls)
    end
    return Jl
end

function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    K = pairwise(metric(kernel),X',Y')
    Jl = zero(K)
    l = getvalue(kernel.lengthscales[1])
    @inbounds for i in eachindex(K)
        Jl[i] = K[i]/l
        K[i] = compute(kernel,K[i])
    end
    return [getvalue(kernel.variance)*Jl.*K,K]
end

"When K has already been computed"
function kernelderivativematrix(X::Array{T,N},Y::Array{T,N},K::Array{T,N},kernel::RBFKernel{T,PlainKernel}) where {T,N}
    P = pairwise(metric(kernel),X',Y')
    Jl = zero(K)
    l = getvalue(kernel.lengthscales[1])
    @inbounds for i in eachindex(K)
        Jl[i] = P[i]/l
    end
    return [Jl.*K,K./getvalue(kernel.variance)]
end



"""
    Create a symmetric kernel matrix from training data
"""
function oldkernelmatrix!(K::Matrix{T},X::Matrix{T},kernel::Kernel{T}) where T
    @assert size(K,1) == size(X,1)
    n = size(K,1)
    return K .= Symmetric([j<=i ? compute(kernel,X[i,:],X[j,:]) : 0.0 for i in 1:n, j in 1:n],:L)
end

function oldkernelmatrix(X::Matrix{T},kernel::Kernel{T}) where T
    n = size(X,1);
    return Symmetric([j<=i ? compute(kernel,X[i,:],X[j,:]) : 0.0 for i in 1:n, j in 1:n],:L)
end

"""
    Only compute the variance (diagonal elements)
"""
function olddiagkernelmatrix!(K::Vector{T},X::Matrix{T},kernel::Kernel{T}) where T
    n = size(K,1)
    return K = [compute(kernel,X[i,:],X[i,:]) for i in 1:n]
end

function olddiagkernelmatrix(X::Matrix{T},kernel::Kernel{T}) where T
    n = size(X,1)
    return [compute(kernel,X[i,:],X[i,:]) for i in 1:n]
end

"""
    Compute derivative of the kernel matrix given kernel hyperparameters
"""
function derivativekernelmatrix(kernel::Kernel{T},X1::Matrix{T},X2::Matrix{T}) where T
    n1 = size(X1,1); n2 = size(X2,1)
    A = [Matrix{Float64}(undef,n1,n2) for _ in 1:(length(kernel.param.hyperparameters)+1)]
    for i in 1:n1, j in 1:n2
        g = compute_deriv(kernel,X1[i,:],X2[j,:],true);
        [a[i,j] = g[iter] for (iter,a) in enumerate(A) ];
    end
    return A
end

function derivativekernelmatrix(kernel::Kernel{T},X::Matrix{T}) where T
    n = size(X,1)
    A = [Matrix{Float64}(undef,n,n) for _ in 1:(kernel.param.Nparam+1)]
    for i in 1:n, j in 1:n
        if i<=j
            g = compute_deriv(kernel,X[i,:],X[j,:],true);
            [a[i,j] = g[iter] for (iter,a) in enumerate(A) ];
        else
            [a[i,j] = a[j,i] for a in A]
        end
    end
    return A
end

function derivativediagkernelmatrix(kernel::Kernel{T},X::Matrix{T}) where T
    n = size(X,1)
    A = [Vector{Float64}(undef,n) for _ in 1:(length(kernel.param.hyperparameters)+1)]
    for i in 1:n
        g = compute_deriv(kernel,X[i,:],X[i,:],true);
        [a[i] = g[iter] for (iter,a) in enumerate(A) ];
        end
    return A
end
# function derivativekernelmatrix(kernel::Kernel{T},X1::Matrix{T},X2::Matrix{T}) where T
#     return compute_J(kernel,compute_unmappedJ(kernel,X1,X2),size(X1,1),size(X2,1))
# end
# function derivativekernelmatrix(kernel::Kernel{T},X::Matrix{T}) where T
#     return compute_J(kernel,compute_unmappedJ(kernel,X),size(X,1),size(X,1))
# end
#
# function derivativediagkernelmatrix(kernel::Kernel{T},X::Matrix{T}) where T
#     return compute_J(kernel,compute_unmappeddiagJ(kernel,X),size(X,1),size(X,1),true,diag=true)
# end


function compute_unmappedJ(kernel::Kernel{T},X1::Matrix{T},X2::Matrix{T}) where T
    n1 = size(X1,1)
    n2 = size(X2,1)
    J = [compute_deriv(kernel,X1[i,:],X2[j,:],true) for i in 1:n1, j in 1:n2]
    return J[:]
end

function compute_unmappedJ(kernel::Kernel{T},X::Matrix{T}) where T
    n = size(X,1)
    J = [compute_deriv(kernel,X[i,:],X[j,:],true) for i in 1:n, j in 1:n]
    return J[:]
end

function compute_unmappeddiagJ(kernel::Kernel{T},X::Matrix{T}) where T
    n = size(X,1)
    return [compute_deriv(kernel,X[i,:],X[i,:],true) for i in 1:n]
end

function compute_J(k::KernelSum{T},J,n1::Int64,n2::Int64,variance::Bool=true;diag::Bool=false) where T
    return [compute_J(kernel,[j[i] for j in J],n1,n2,true,diag=diag) for (i,kernel) in enumerate(k.kernel_array)]
end

function compute_J(k::KernelProduct{T},J,n1::Int64,n2::Int64,variance::Bool=true;diag::Bool=false) where T
    J_mat = Vector{Any}(undef,k.Nkernels+variance)
    for (i,kernel) in enumerate(k.kernel_array)
        J_mat[i] = compute_J(kernel,[j[i] for j in J],n1,n2,false,diag=diag)
    end
    if variance
        J_mat[end] = diag ? [j[end][1] for j in J] : [reshape([j[end][1] for j in J],n1,n2)]
    end
    return J_mat
end

function compute_J(k::Kernel{T},J,n1::Int64,n2::Int64,variance::Bool=true;diag::Bool=false) where T
    J_mat = diag ? [Vector{Float64}(undef,n1) for i in 1:(k.Nparam+variance)] : [Matrix{Float64}(undef,n1,n2) for i in 1:(k.Nparam+variance)]
    for i in 1:k.Nparam
        if diag
            J_mat[i] .= [j[i] for j in J]
        else
            J_mat[i] .= reshape([j[i] for j in J],n1,n2)
        end
    end
    if variance
        if diag
            J_mat[end] .= [j[end] for j in J]
        else
            J_mat[end] .= reshape([j[end] for j in J],n1,n2)
        end
    end
    return J_mat
end

"""
    Compute the gradients using a gradient function and matrices Js
"""
function compute_hyperparameter_gradient(k::KernelSum{T},gradient_function::Function,variance::Bool,Js::Vector{Any},Kindex::Int64,index::Int64) where T
    return [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
end

function compute_hyperparameter_gradient(k::KernelProduct{T},gradient_function::Function,variance::Bool,Js::Vector{Any},Kindex::Int64,index::Int64) where T
    gradients = [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
    if variance
        push!(gradients,[gradient_function(broadcast(x->x[end][1],Js),Kindex,index)])
    end
    return gradients
end
function compute_hyperparameter_gradient(k::Kernel{T},gradient_function::Function,variance::Bool,Js::Array{Array{T2,1} where T2,1},Kindex::Int64,index::Int64) where T
    return [gradient_function(broadcast(x->x[j],Js),Kindex,index) for j in 1:(k.Nparam+variance)]
end


"""
    Compute derivative matrices given the data points
"""
function CreateColumnRowMatrix(n,iter,gradient)
    K = zeros(n,n)
    K[iter,:] = gradient; K[:,iter] = gradient;
    return K
end

function CreateColumnMatrix(n,m,iter,gradient)
    K = zeros(n,m)
    K[:,iter] = gradient;
    return K
end

#Compute the gradients given the inducing point locations
function computeIndPointsJ(model,iter)
    Dnm = zeros(model.nSamplesUsed,model.nDim)
    Dmm = zeros(model.m,model.nDim)
    Jnm = zeros(model.nDim,model.nSamplesUsed,model.m)
    Jmm = zeros(model.nDim,model.m,model.m)
    #Compute the gradients given every data point
    for i in 1:model.nSamplesUsed
        Dnm[i,:] = compute_point_deriv(model.kernel,model.X[model.MBIndices[i],:],model.inducingPoints[iter,:])
    end
    for i in 1:model.m
        Dmm[i,:] = compute_point_deriv(model.kernel,model.inducingPoints[iter,:],model.inducingPoints[i,:])
    end
    for i in 1:model.nDim
        Jnm[i,:,:] = CreateColumnMatrix(model.nSamplesUsed,model.m,iter,Dnm[:,i])
        Jmm[i,:,:] = CreateColumnRowMatrix(model.m,iter,Dmm[:,i])
    end
    return Jnm,Jmm
    #Return dim * K*K tensors for computing the gradient
end
