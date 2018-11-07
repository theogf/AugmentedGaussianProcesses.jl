function AGP(X::AbstractArray{T,N},y::AbstractArray{T2,N2},likelihood::Symbol;Sparse::Bool=true,Stochastic::Bool=true)
    if likelihood == :Gaussian
        if Sparse
            return SparseGPRegression(X,y)
        else
            return BatchGPRegression(X,y)
        end
    elseif likelihood == :StudentT
        if Sparse
            return SparseStudentT(X,y)
        else
            return BatchStudentT(X,y)
        end
    elseif likelihood == :Logistic
        if Sparse
            return SparseXGPC(X,y)
        else
            return BatchXGPC(X,y)
        end
    elseif likelihood == :BSVM
        if Sparse
            return SparseBSVM(X,y)
        else
            return BatchBSVM(X,y)
        end
    else
        error("Likelihood not valid, options for regression are:\n\t :Gaussian, :StudentT\n options for classification are:\n\t:Logistic, :BSVM")
    end

end
