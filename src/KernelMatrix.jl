"""
    Create the kernel matrix from the training data or the correlation matrix one of set of vectors
"""
function kernelmatrix!(K,X1,X2,kernel_function)
    (n1,n2) = size(K)
    for i in 1:n1
      for j in 1:n2
        K[i,j] = kernel_function(X1[i,:],X2[j,:])
      end
    end
    return K
end
function kernelmatrix(X1,X2,kernel_function)
    n1 = size(X1,1)
    n2 = size(X2,1)
    K = zeros(n1,n2)
    return CreateKernelMatrix!(K,X1,X2,kernel_function)
end
"""
    Create the kernel matrix from the training data or the correlation matrix between two set of vectors
"""
function kernelmatrix!(K,X,kernel_function)
    n = size(K,1)
    for i in 1:n
      for j in 1:i
        K[i,j] = kernel_function(X[i,:],X[j,:])
      end
    end
    return Symmetric(K,uplo=:L)
end
function kernelmatrix(X,kernel_function)
    n = size(X,1);
    K = zeros(n,n);
    return CreateKernelMatrix!(K,X,kernel_function)
end

"""
    Only compute the variance (diagonal elements)
"""
function CreateDiagonalKernelMatrix(X,kernel_function)
    n = size(X,1)
    k = zeros(n)
    return CreateDiagonalKernelMatrix(k,X,kernel_function)
end

function CreateDiagonalKernelMatrix!(k,X,kernel_function)
    n = size(k,1)
    for i in 1:n
        k[i] = kernel_function(X[i,:],X[i,:])
    end
    return k
end
