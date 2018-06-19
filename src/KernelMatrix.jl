"""
    Create the kernel matrix from the training data or the correlation matrix one of set of vectors
"""
function kernelmatrix!(K,X1,X2,kernel)
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
function kernelmatrix(X1,X2,kernel)
    n1 = size(X1,1)
    n2 = size(X2,1)
    K = zeros(n1,n2)
    return kernelmatrix!(K,X1,X2,kernel)
end
"""
    Create a symmetric kernel matrix from training data
"""
function kernelmatrix!(K,X,kernel)
    @assert size(K,1) == size(X,1)
    n = size(K,1)
    for i in 1:n
      for j in 1:i
        K[i,j] = compute(kernel,X[i,:],X[j,:])
      end
    end
    return Symmetric(K,:L)
end
function kernelmatrix(X,kernel)
    n = size(X,1);
    K = zeros(n,n);
    return kernelmatrix!(K,X,kernel)
end

"""
    Only compute the variance (diagonal elements)
"""

function diagkernelmatrix!(k,X,kernel)
    n = size(k,1)
    for i in 1:n
        k[i] = compute(kernel,X[i,:],X[i,:])
    end
    return k
end

function diagkernelmatrix(X,kernel)
    n = size(X,1)
    k = zeros(n)
    return diagkernelmatrix!(k,X,kernel)
end

"""
    Compute derivative of the kernel matrix given kernel hyperparameters
"""


function derivativekernelmatrix(kernel,X1,X2)
    return compute_J(kernel,compute_unmappedJ(kernel,X1,X2),size(X1,1),size(X2,1))
end

function derivativekernelmatrix(kernel,X)
    return compute_J(kernel,compute_unmappedJ(kernel,X),size(X,1),size(X,1))
end

function derivativediagkernelmatrix(kernel,X)
    return compute_J(kernel,compute_unmappeddiagJ(kernel,X),size(X,1),size(X,1),weight=true,diag=true)
end


function compute_unmappedJ(kernel,X1,X2)
    n1 = size(X1,1)
    n2 = size(X2,1)
    J = Array{Any,2}(n1,n2)
    for i in 1:n1
        for j in 1:n2
            J[i,j] = compute_deriv(kernel,X1[i,:],X2[j,:],true)
        end
    end
    return J[:]
end

function compute_unmappedJ(kernel,X)
    n = size(X,1)
    J = Array{Any,2}(n,n)
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

function compute_unmappeddiagJ(kernel,X)
    n = size(X,1)
    J = Array{Any,1}(n)
    for i in 1:n
        J[i] = compute_deriv(kernel,X[i,:],X[i,:],true)
    end
    return J
end

function compute_J(k::KernelSum,J,n1,n2,weight::Bool=true;diag::Bool=false)
    J_mat = Array{Any,1}()
    for (i,kernel) in enumerate(k.kernel_array)
        push!(J_mat,compute_J(kernel,broadcast(x->x[i],J),n1,n2,true,diag=diag))
    end
    return J_mat
end

function compute_J(k::KernelProduct,J,n1,n2,weight::Bool=true;diag::Bool=false)
    J_mat = Array{Any,1}()
    for (i,kernel) in enumerate(k.kernel_array)
        push!(J_mat,compute_J(kernel,broadcast(x->x[i],J),n1,n2,false,diag=diag))
    end
    if weight
        if diag
            push!(J_mat,broadcast(x->x[end][1],J))
        else
            push!(J_mat,[reshape(broadcast(x->x[end][1],J),n1,n2)])
        end
    end
    return J_mat
end

function compute_J(k::Kernel,J,n1,n2,weight::Bool=true;diag::Bool=false)
    J_mat = Array{Any,1}()
    for i in k.Nparam
        if diag
            push!(J_mat,broadcast(x->x[i],J))
        else
            push!(J_mat,reshape(broadcast(x->x[i],J),n1,n2))
        end
    end
    if weight
        if diag
            push!(J_mat,broadcast(x->x[end],J))
        else
            push!(J_mat,reshape(broadcast(x->x[end],J),n1,n2))
        end
    end
    return J_mat
end

"""
    Compute the gradients using a gradient function and matrices Js
"""


function compute_hyperparameter_gradient(k::KernelSum,gradient_function::Function,Js,weight::Bool=true)
    gradients = Array{Any,1}()
    for (i,kernel) in enumerate(k.kernel_array)
        push!(gradients,compute_hyperparameter_gradient(kernel,gradient_function,broadcast(x->x[i],Js),true))
    end
    return gradients
end

function compute_hyperparameter_gradient(k::KernelProduct,gradient_function::Function,Js,weight::Bool=true)
    gradients = Array{Any,1}()
    for (i,kernel) in enumerate(k.kernel_array)
        push!(gradients,compute_hyperparameter_gradient(kernel,gradient_function,broadcast(x->x[i],Js),false))
    end
    if weight
        push!(gradients,[gradient_function(broadcast(x->x[end][1],Js))])
    end
    return gradients
end
function compute_hyperparameter_gradient(k::Kernel,gradient_function::Function,Js,weight::Bool=true)
    gradients = Array{Float64,1}()
    for i in k.Nparam
        push!(gradients,gradient_function(broadcast(x->x[i],Js)))
    end
    if weight
        push!(gradients,gradient_function(broadcast(x->x[end],Js)))
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
    dim = size(model.X,2)
    Dnm = zeros(model.nSamplesUsed,dim)
    Dmm = zeros(model.m,dim)
    Jnm = zeros(dim,model.nSamplesUsed,model.m)
    Jmm = zeros(dim,model.m,model.m)
    #Compute the gradients given every other point
    for i in 1:model.nSamplesUsed
        Dnm[i,:] = compute_point_deriv(model.X[model.MBIndices[i],:],model.inducingPoints[iter,:],model.kernel)
    end
    for i in 1:model.m
        Dmm[i,:] = compute_point_deriv(model.inducingPoints[iter,:],model.inducingPoints[i,:],model.kernel)
    end
    for i in 1:dim
        Jnm[i,:,:] = CreateColumnMatrix(model.nSamplesUsed,model.m,iter,Dnm[:,i],model.kernel)
        Jmm[i,:,:] = CreateColumnRowMatrix(model.m,iter,Dmm[:,i],model.kernel)
    end
    return Jnm,Jmm
end
