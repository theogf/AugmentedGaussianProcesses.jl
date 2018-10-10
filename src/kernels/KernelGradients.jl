"""
    Compute the gradients using a gradient function and matrices Js
"""

#Case for full model with ARD Kernel
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,J::Vector{LinearAlgebra.Symmetric{Float64,Matrix{Float64}}}) where T
    return map(gradient_function,J)
end

#Case for sparse with ARD Kernel
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,Js::Vector{Array{T2,1} where T2}) where T
    return map(gradient_function,Js[1],Js[2],Js[3])
end

#Case for full batch with Plain Kernel
function compute_hyperparameter_gradient(k::Kernel{T,PlainKernel},gradient_function::Function,J::LinearAlgebra.Symmetric{Float64,Matrix{Float64}}) where T
    return gradient_function(J)
end

#Case for sparse with Plain Kernel
function compute_hyperparameter_gradient(k::Kernel{T,PlainKernel},gradient_function::Function,Js::Vector{AbstractArray{Float64,N} where N}) where T
    return gradient_function(Js[1],Js[2],Js[3])
end


#Case for multiscale full model with ARD Kernel
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,J::Vector{LinearAlgebra.Symmetric{Float64,Matrix{Float64}}},Kindex::Int64,index::Int64) where T
    return map(gradient_function,J,Kindex*ones(Int64,k.fields.Ndim),index*ones(Int64,k.fields.Ndim))
end

#Case for multiscale sparse with ARD Kernel
function compute_hyperparameter_gradient(k::Kernel{T,ARDKernel},gradient_function::Function,Js::Vector{Array{T2,1} where T2},Kindex::Int64,index::Int64) where T
    return map(gradient_function,Js[1],Js[2],Js[3],Kindex*ones(Int64,k.fields.Ndim),index*ones(Int64,k.fields.Ndim))
end

#Case for multiscale full batch with Plain Kernel
function compute_hyperparameter_gradient(k::Kernel{T,PlainKernel},gradient_function::Function,J::LinearAlgebra.Symmetric{Float64,Matrix{Float64}},Kindex::Int64,index::Int64) where T
    return gradient_function(J,Kindex,index)
end

#Case for multiscale sparse with Plain Kernel
function compute_hyperparameter_gradient(k::Kernel{T,PlainKernel},gradient_function::Function,Js::Vector{AbstractArray{Float64,N} where N},Kindex::Int64,index::Int64) where T
    return gradient_function(Js[1],Js[2],Js[3],Kindex,index)
end


# function compute_hyperparameter_gradient(k::KernelSum{T},gradient_function::Function,Js::Vector{Any},Kindex::Int64,index::Int64) where T
#     return [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
# end
#
# function compute_hyperparameter_gradient(k::KernelProduct{T},gradient_function::Function,Js::Vector{Any},Kindex::Int64,index::Int64) where T
#     gradients = [compute_hyperparameter_gradient(kernel,gradient_function,false,broadcast(x->x[j],Js),Kindex,index) for (j,kernel) in enumerate(k.kernel_array)]
#     if variance
#         push!(gradients,[gradient_function(broadcast(x->x[end][1],Js),Kindex,index)])
#     end
#     return gradients
# end
