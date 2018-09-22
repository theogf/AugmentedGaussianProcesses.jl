"""
    Create the kernel matrix from the training data or the correlation matrix one of set of vectors
"""
function kernelmatrix!(K::Matrix{Float64},X1,X2,kernel)
    @assert size(K,1)==size(X1,1)
    @assert size(K,2)==size(X2,1)
    (n1,n2) = size(K)
    for i in 1:n1
      for j in 1:n2
        K[i,j] = compute(kernel,X1[i,:],X2[j,:])
      end
    end
    return K
end
function kernelmatrix(X1,X2,kernel::Kernel)
    n1 = size(X1,1)
    n2 = size(X2,1)
    K = zeros(n1,n2)
    return kernelmatrix!(K,X1,X2,kernel)
end

"""
    Create a symmetric kernel matrix from training data
"""
function kernelmatrix!(K,X::Matrix{Float64},kernel::Kernel)
    @assert size(K,1) == size(X,1)
    n = size(K,1)
    for i in 1:n
      for j in 1:i
        K[i,j] = compute(kernel,X[i,:],X[j,:])
      end
    end
    return Symmetric(K,:L)
end
function kernelmatrix(X::Matrix{Float64},kernel::Kernel)
    n = size(X,1);
    K = zeros(n,n);
    return kernelmatrix!(K,X,kernel)
end

"""
    Only compute the variance (diagonal elements)
"""
function diagkernelmatrix!(k,X::Matrix{Float64},kernel::Kernel)
    n = size(k,1)
    for i in 1:n
        k[i] = compute(kernel,X[i,:],X[i,:])
    end
    return k
end

function diagkernelmatrix(X::Matrix{Float64},kernel::Kernel)
    n = size(X,1)
    k = zeros(n)
    return diagkernelmatrix!(k,X,kernel)
end

"""
    Compute derivative of the kernel matrix given kernel hyperparameters
"""
function derivativekernelmatrix(kernel::Kernel,X1::Matrix{Float64},X2::Matrix{Float64})
    return compute_J(kernel,compute_unmappedJ(kernel,X1,X2),size(X1,1),size(X2,1))
end

function derivativekernelmatrix(kernel::Kernel,X::Matrix{Float64})
    return compute_J(kernel,compute_unmappedJ(kernel,X),size(X,1),size(X,1))
end

function derivativediagkernelmatrix(kernel::Kernel,X::Matrix{Float64})
    return compute_J(kernel,compute_unmappeddiagJ(kernel,X),size(X,1),size(X,1),true,diag=true)
end


function compute_unmappedJ(kernel::Kernel,X1::Matrix{Float64},X2::Matrix{Float64})
    n1 = size(X1,1)
    n2 = size(X2,1)
    J = Array{Any,2}(undef,n1,n2)
    for i in 1:n1
        for j in 1:n2
            J[i,j] = compute_deriv(kernel,X1[i,:],X2[j,:],true)
        end
    end
    return J[:]
end

function compute_unmappedJ(kernel::Kernel,X::Matrix{Float64})
    n = size(X,1)
    J = Matrix{Any}(undef,n,n)
    for i in 1:n
        for j in 1:i
            J[i,j] = compute_deriv(kernel,X[i,:],X[j,:],true)
            if i!=j
                J[j,i] = J[i,j]
            end
        end
    end
    return J[:]
end

function compute_unmappeddiagJ(kernel::Kernel,X::Matrix{Float64})
    n = size(X,1)
    J = Vector{Any}(undef,n)
    for i in 1:n
        J[i] = compute_deriv(kernel,X[i,:],X[i,:],true)
    end
    return J
end

function compute_J(k::KernelSum,J,n1::Int64,n2::Int64,variance::Bool=true;diag::Bool=false)
    J_mat = Vector{Any}(undef,k.Nkernels)
    for (i,kernel) in enumerate(k.kernel_array)
        J_mat[i] = compute_J(kernel,[j[i] for j in J],n1,n2,true,diag=diag)
    end
    return J_mat
end

function compute_J(k::KernelProduct,J,n1::Int64,n2::Int64,variance::Bool=true;diag::Bool=false)
    J_mat = Vector{Any}(undef,k.Nkernels+variance)
    for (i,kernel) in enumerate(k.kernel_array)
        J_mat[i] = compute_J(kernel,[j[i] for j in J],n1,n2,false,diag=diag)
    end
    if variance
        J_mat[end] = diag ? [j[end][1] for j in J] : [reshape([j[end][1] for j in J],n1,n2)]
    end
    return J_mat
end

function compute_J(k::Kernel,J,n1::Int64,n2::Int64,variance::Bool=true;diag::Bool=false)
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
function compute_hyperparameter_gradient(k::KernelSum,gradient_function::Function,variance::Bool,Js,Kindex,index)
    gradients = Array{Any,1}()
    for (j,kernel) in enumerate(k.kernel_array)
        push!(gradients,compute_hyperparameter_gradient(kernel,gradient_function,true,broadcast(x->x[j],Js),Kindex,index))
    end
    return gradients
end

function compute_hyperparameter_gradient(k::KernelProduct,gradient_function::Function,variance::Bool,Js,Kindex,index)
    gradients = Array{Any,1}()
    for (j,kernel) in enumerate(k.kernel_array)
        push!(gradients,compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index))
    end
    if variance
        push!(gradients,[gradient_function(broadcast(x->x[end][1],Js),Kindex,index)])
    end
    return gradients
end
function compute_hyperparameter_gradient(k::Kernel,gradient_function::Function,variance::Bool,Js,Kindex,index)
    gradients = Array{Float64,1}()
    for j in 1:k.Nparam
        push!(gradients,gradient_function(broadcast(x->x[j],Js),Kindex,index))
    end
    if variance
        push!(gradients,gradient_function(broadcast(x->x[end],Js),Kindex,index))
    end
    return gradients
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
