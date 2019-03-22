"Structure to combine kernels together by multiplication, can be created by using the constructor with an array or simply using Base.*"
mutable struct KernelProduct{T<:AbstractFloat,KernelCombination} <: Kernel{T,KernelCombination}
    # @kernelfunctionfields
    kernel_array::Vector{Kernel} #Array of multiplied kernels
    Nkernels::Int64 #Number of multiplied kernels
    "Inner KernelProduct constructor taking an array of kernels"
    function KernelProduct{T}(kernels::Vector{Kernel{T}}) where {T<:AbstractFloat}
        this = new{T,KernelCombination}("Product of kernels",
                HyperParameter{T}(1.0,interval(OpenBound(zero(T)),nothing),fixed=false),
                HyperParameters{T}(),
                0,
                HyperParameters{T}()
                )
        this.kernel_array = deepcopy(Vector{Kernel}(kernels))
        this.Nkernels = length(this.kernel_array)
        this.distance = Identity
        return this
    end
end

"Apply kernel functions on vector"
function compute(k::KernelProduct{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    product = 1.0
    for kernel in k.kernel_array
        product *= compute(kernel,X1,X2,false)
    end
    return (variance ? getvalue(k.variance) : 1.0) * product
end

"Compute kernel gradients given the vectors"
function compute_deriv(k::KernelProduct{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    tot = compute(k,X1,X2)
    deriv_values = Vector{Vector{Any}}()
    for kernel in k.kernel_array
        push!(deriv_values,compute_deriv(kernel,X1,X2,false).*tot./compute(kernel,X1,X2,false))
    end
    if variance
        push!(deriv_values,[tot])
    end
    return deriv_values
end

function compute_point_deriv(k::KernelProduct{T},X1::Vector{T},X2::Vector{T}) where T
    tot = compute(k,X1,X2)
    deriv_values = zeros(X1)
    for kernel in k.kernel_array
        #This should be checked TODO
        deriv_values += compute_point_deriv(kernel,X1,X2).*tot./compute(kernel,X1,X2)
    end
    return deriv_values
end

function Base.:*(a::Kernel{T},b::Kernel{T}) where T
    return KernelProduct{T}([a,b])
end
function Base.:*(a::KernelProduct{T},b::Kernel{T}) where T
    return KernelProduct{T}(vcat(a.kernel_array,b))
end
function Base.:*(a::KernelProduct{T},b::KernelSum{T}) where T
    return KernelProduct{T}(vcat(a.kernel_array,b.kernel_array))
end
function Base.getindex(a::KernelProduct{T},i::Int64) where T
    return a.kernel_array[i]
end
