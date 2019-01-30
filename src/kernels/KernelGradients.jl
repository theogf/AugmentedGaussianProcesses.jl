### Compute the gradients using a gradient function and matrices Js ###

"""Gradient computation for full model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,J::Vector{LinearAlgebra.Symmetric{Float64,Matrix{Float64}}}) where T
    return map(gradient_function,J)
end

"""Gradient computation for sparse model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,Js::Vector{Array{T2,1} where T2}) where T
    return map(gradient_function,Js[1],Js[2],Js[3])
end

"""Gradient computation for full model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,J::LinearAlgebra.Symmetric{Float64,Matrix{Float64}}) where T
    return gradient_function(J)
end

"""Gradient computation for sparse model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,Js::Vector{AbstractArray{Float64,N} where N}) where T
    return gradient_function(Js[1],Js[2],Js[3])
end


"""Gradient computations for full multiclass model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,J::Vector{LinearAlgebra.Symmetric{Float64,Matrix{Float64}}},index::Int64) where T
    return map(gradient_function,J,index*ones(Int64,k.fields.Ndim))
end

"""Gradient computations for sparse multiclass model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,Js::Vector{Array{T2,1} where T2},index::Int64) where T
    return map(gradient_function,Js[1],Js[2],Js[3],index*ones(Int64,k.fields.Ndim))
end

"""Gradient computations for full multiclass model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,J::LinearAlgebra.Symmetric{Float64,Matrix{Float64}},index::Int64) where T
    return gradient_function(J,index)
end

"""Gradient computations for sparse multiclass model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,Js::Vector{AbstractArray{Float64,N} where N},index::Int64) where T
    return gradient_function(Js[1],Js[2],Js[3],index)
end
