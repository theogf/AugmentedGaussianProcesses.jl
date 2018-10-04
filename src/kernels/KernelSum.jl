"Structure to combine kernels together by addition, can be created by using the constructor with an array or simply using Base.+"
mutable struct KernelSum{T<:AbstractFloat,KernelCombination} <: Kernel{T,KernelCombination}
    @kernelfunctionfields()
    kernel_array::Vector{Kernel} #Array of summed kernels
    Nkernels::Int64 #Number of kernels
    "Inner KernelSum constructor taking an array of kernels"
    function KernelSum{T,KernelCombination}(kernels::AbstractArray) where {T<:AbstractFloat}
        this = new("Sum of kernels")
        this.kernel_array = deepcopy(kernels)
        this.Nkernels = length(this.kernel_array)
        this.distance = Identity
        return this
    end
end

"Apply kernel functions on vector"
function compute(k::KernelSum{T,KernelCombination},X1::Vector{T},X2::Vector{T},variance::Bool=true) where {T}
    sum = 0.0
    for kernel in k.kernel_array
        sum += compute(kernel,X1,X2)
    end
    return sum
end

"Compute kernel gradients given the vectors"
function compute_deriv(k::KernelSum{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where {T}
    deriv_values = Vector{Vector{Any}}()
    for kernel in k.kernel_array
        push!(deriv_values,compute_deriv{T}(kernel,X1,X2,true))
    end
    return deriv_values
end

function compute_point_deriv(k::KernelSum{T},X1,X2) where {T}
    deriv_values = zeros(X1)
    for kernel in k.kernel_array
        deriv_values += getvalue(kernel.variance)*compute_point_deriv{T}(kernel,X1,X2)
    end
    return deriv_values
end

function Base.:+(a::Kernel{T},b::Kernel{T}) where T
    return KernelSum{T}([a,b])
end
function Base.:+(a::KernelSum{T},b::Kernel{T}) where T
    return KernelSum{T}(vcat(a.kernel_array,b))
end
function Base.:+(a::Kernel{T},b::KernelSum{T}) where T
    return KernelSum{T}(vcat(a,b.kernel_array))
end
function Base.:+(a::KernelSum{T},b::KernelSum{T}) where T
    return KernelSum{T}(vcat(a.kernel_array,b.kernel_array))
end

"Return the kernel at index i of the kernel sum"
function Base.getindex(a::KernelSum{T},i::Int64) where T
    return a.kernel_array[i]
end
