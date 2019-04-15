abstract type KernelType end;

abstract type ARDKernel <: KernelType end;

abstract type IsoKernel <: KernelType end;

abstract type KernelCombination <: KernelType end;


"Macro giving common fields for all kernels"
mutable struct KernelFields{T,KT}
    name::String #Name of the kernel function
    variance::HyperParameter{T} #variance of the kernel
    lengthscales::HyperParameters{T} #lengthscale variables
    Ndim::Int64 #Number of lengthscales
    metric::PreMetric
end

# params::HyperParameters{T} #Parameters of the kernel
# Nparam::Int64 #Number of parameters
"""Abstract type for all kernels"""
abstract type Kernel{T<:Real, KT<:KernelType} end;

"""Return the metric of a kernel"""
function getmetric(k::Kernel{T,KT}) where {T<:Real,KT<:KernelType}
    return k.fields.metric
end

"""Return the variance of the kernel"""
function getvariance(k::Kernel{T,KT}) where {T<:Real,KT<:KernelType}
    return getvalue(k.fields.variance)
end

"""Return the lengthscale of the IsoKernel"""
function getlengthscales(k::Kernel{T,IsoKernel}) where {T<:Real}
    return getvalue(k.fields.lengthscales[1])
end

"""Return the lengthscales of the ARD Kernel"""
function getlengthscales(k::Kernel{T,ARDKernel}) where {T<:Real}
    return getvalue(k.fields.lengthscales)
end

"""Set optimizers of all parameters of the kernel"""
function setoptimizer!(k::Kernel{T},opt::Optimizer) where {T<:Real}
    setparamoptimizer!(k.fields.variance,opt)
    setparamoptimizer!(k.fields.lengthscales,opt)
end

"""Get optimizer for variance of kernel"""
function getvarianceoptimizer(k::Kernel{T}) where {T<:Real}
    k.fields.variance.opt
end

isARD(::Kernel{T,KT}) where {T<:Real,KT<:ARDKernel} = true
isARD(::Kernel{T,KT}) where {T<:Real,KT<:IsoKernel} = false
isIso(::Kernel{T,KT}) where {T<:Real,KT<:ARDKernel} = false
isIso(::Kernel{T,KT}) where {T<:Real,KT<:IsoKernel} = true

function Base.show(io::IO,k::Kernel{T,KT}) where {T,KT}
    print(io,"$(k.fields.name)"*(isARD(k) ? " ARD" : "")*" kernel, with variance $(getvariance(k)) and lengthscales $(getlengthscales(k))")
end

"""Standard conversion when giving scalar and not vectors"""
function compute(k::Kernel{T,KT},X1::T,X2::T) where {T<:Real,KT<:KernelType}
    compute(k,[X1],[X2])
end

"""Function to determine most adapted type between a selection"""
function floattype(T_i::DataType...)
    T_max = promote_type(T_i...)
    T_max <: Real ? T_max : Float64
end

function convert(::Type{K},x::Kernel{T2,KT}) where {K<:Kernel{T} where T<:Real,T2<:Real,KT<:KernelType}

end

function apply_gradients_lengthscale!(kernel::Kernel{T,IsoKernel},gradient::T) where {T}
    update!(kernel.fields.lengthscales,[gradient])
end

function apply_gradients_lengthscale!(kernel::Kernel{T,ARDKernel},gradients::Vector{T}) where {T}
    update!(kernel.fields.lengthscales,gradients)
    updateweights!(kernel,getlengthscales(kernel))
end

function apply_gradients_variance!(kernel::Kernel{T,KT},gradient::T) where {T,KT}
    update!(kernel.fields.variance,gradient)
end
#
# function apply_gradients!(kernel::KernelSum,gradients,variance::Bool=true)
#     for i in 1:kernel.Nkernels
#         apply_gradients!(kernel.kernel_array[i],gradients[i],true)
#     end
# end
#
# function apply_gradients!(kernel::KernelProduct,gradients,variance::Bool=true)
#     for i in 1:kernel.Nkernels
#         apply_gradients!(kernel.kernel_array[i],gradients[i],false);
#     end
#     if variance
#         update!(kernel.variance,gradients[end][1]);
#     end
# end
