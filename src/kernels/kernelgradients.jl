### Compute the gradients using a gradient function and matrices Js ###

"""Gradient computation for full model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,J::Vector{LinearAlgebra.Symmetric{Float64,Matrix{Float64}}}) where {T<:Real}
    return map(gradient_function,J)
end

"""Gradient computation for sparse model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,Jmm::Vector{Symmetric{T,Matrix{T}}},Jnm::Vector{Matrix{T}}, Jnn::Vector{Vector{T}}) where T
    return map(gradient_function,Jmm,Jnm,Jnn)
end

"""Gradient computation for full model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,J::Symmetric{Float64,Matrix{Float64}}) where T
    return gradient_function(J)
end

"""Gradient computation for sparse model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,Jmm::Vector{Symmetric{T,Matrix{T}}},Jnm::Vector{Matrix{T}}, Jnn::Vector{Vector{T}}) where T
    return gradient_function(Jmm,Jnm,Jnn)
end


"""Gradient computations for full multiclass model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,J::Vector{Symmetric{T,Matrix{T}}},index::Int64) where T
    return map(gradient_function,J,fill(index,k.fields.Ndim))
end

"""Gradient computations for sparse multiclass model with ARD Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,Jmm::Vector{Symmetric{T,Matrix{T}}},Jnm::Vector{Matrix{T}}, Jnn::Vector{Vector{T}},index::Int64) where T
    return map(gradient_function,Jmm,Jnm,Jnn,fill(index,k.fields.Ndim))
end

"""Gradient computations for full multiclass model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,J::Symmetric{T,Matrix{T}},index::Int64) where T
    return gradient_function(J,index)
end

"""Gradient computations for sparse multiclass model with Iso Kernel"""
function compute_hyperparameter_gradient(k::Kernel{T,IsoKernel},gradient_function::Function,Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},index::Int64) where T
    return gradient_function(Jmm,Jnm,Jnn,index)
end
