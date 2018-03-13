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
    return CreateKernelMatrix!(K,X1,X2,kernel)
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
    return Symmetric(K,uplo=:L)
end
function kernelmatrix(X,kernel)
    n = size(X,1);
    K = zeros(n,n);
    return CreateKernelMatrix!(K,X,kernel)
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
    Compute derivative of the kernel matrix
"""

function update_kernel_hyperparameters!(model::FullBatchModel)
    Jnn = compute_J(model.kernel,compute_unmappedJ(model.kernel,model.X),model.nSamples,model.nSamples)
    apply_gradients!(model.kernel,compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jnn]))
end
function update_kernel_hyperparameters!(model::SparseModel)
    Jmm = compute_J(model.kernel,compute_unmappedJ(model.kernel,model.inducingPoints),model.m,model.m)
    Jmm = compute_J(model.kernel,compute_unmappedJ(model.kernel,model.X[model.MBIndices],model.inducingPoints),model.nSamplesUsed,model.m)
    Jnn = compute_J(model.kernel,compute_unmappeddiagJ(model.kernel,model.X[model.MBIndices]),model.nSamplesUsed,diag=true)
    apply_gradients!(model.kernel,compute_hyperparameter_gradient(model.kernel,hyperparameter_gradient_function(model),Any[Jnn,Jnm,Jnn]))
end

function compute_unmappedJ(kernel,X1,X2)
    n1 = size(X1,1)
    n2 = size(X2,1)
    J = Array{Any,2}(n1,n2)
    for i in 1:n1
        for j in 1:n2
            J[i,j] = compute_deriv(kernel,X[i,:],Y[j,:],true)
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
        J[i,i] = compute_deriv(kernel,X[i,:],X[i,:],true)
    end
    return J[:]
end

function compute_J(k::KernelSum,J,n1,n2,weight::Bool=true,diag::Bool=false)
    J_mat = Array{Any,1}()
    for (i,kernel) in enumerate(k.kernel_array)
        push!(J_mat,compute_J(kernel,broadcast(x->x[i],J),n1,n2,true))
    end
    return J_mat
end

function compute_J(k::KernelProduct,J,n1,n2,weight::Bool=true,diag::Bool=false)
    J_mat = Array{Any,1}()
    for (i,kernel) in enumerate(k.kernel_array)
        push!(J_mat,compute_J(kernel,broadcast(x->x[i],J),n1,n2,false))
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

function compute_J(k::Kernel,J,n1,n2,weight::Bool=true,diag::Bool=false)
    J_mat = Array{Any,1}()
    for i in k.Nparameters
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
    for i in k.Nparameters
        push!(gradients,gradient_function(broadcast(x->x[i],Js)))
    end
    if weight
        push!(gradients,gradient_function(broadcast(x->x[end],Js)))
    end
    return gradients
end
