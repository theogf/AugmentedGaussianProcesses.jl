abstract type AbstractKernelWrapper end

mutable struct KernelWrapper{K<:Kernel} <: AbstractKernelWrapper
    kernel::K
    params
    opts
end

params(k::KernelWrapper) = k.params
KernelFunctions.set_params!(k::KernelWrapper,θ::AbstractVector) = k.params .= θ

mutable struct KernelSumWrapper{T<:Real,K<:AbstractKernelWrapper} <: AbstractKernelWrapper
    kernelwrappers::Vector{K}
    weights::Vector{T}
    opt::OptorNothing
end

Base.getindex(k::KernelSumWrapper,i::Int) = k.kernelwrappers[i]
Base.length(k::KernelSumWrapper) = length(k.kernelwrappers)
Base.iterate(k::KernelSumWrapper) = iterate(k.kernelwrappers)
Base.iterate(k::KernelSumWrapper,state) = iterate(k.kernelwrappers,state)

mutable struct KernelProductWrapper{T} <: AbstractKernelWrapper
    kernelwrappers::Vector{AbstractKernelWrapper}
end

Base.getindex(k::KernelProductWrapper,i::Int) = k.kernelwrappers[i]
Base.length(k::KernelProductWrapper) = length(k.kernelwrappers)
Base.iterate(k::KernelProductWrapper) = iterate(k.kernelwrappers)
Base.iterate(k::KernelProductWrapper,state) = iterate(k.kernelwrappers,state)

function wrapper(kernel::Kernel,opt::OptorNothing)
    p = collect(KernelFunctions.params(kernel))
    opts = create_opts(kernel,opt)
    KernelWrapper(kernel,p,opts)
end


function wrapper(kernel::KernelSum,opt::OptorNothing)
    KernelSumWrapper(wrapper.(kernel.kernels,[opt]),kernel.weights,deepcopy(opt))
end

function wrapper(kernel::KernelProduct,opt::OptorNothing)
    KernelProductWrapper(wrapper.(kernel,[opt]))
end

isopt(k::KernelWrapper) = count(!isnothing,k.opts) > 0
isopt(k::KernelSumWrapper) = !isnothing(k.opt) || count(isopt,k) > 0
isopt(k::KernelProductWrapper) = count(isopt,k) > 0

function create_opts(kernel::Kernel,opt::OptorNothing)
    opts = []
    for p in KernelFunctions.opt_params(kernel)
        push!(opts,isnothing(p) ? nothing : deepcopy(opt))
    end
    return opts
end

function create_opts(kernel::Kernel{T,<:ChainTransform},opt::OptorNothing) where {T}
    opts = []
    t_opts = []
    ps = KernelFunctions.opt_params(kernel)
    for p in first(ps)
        @show p
        push!(t_opts,isnothing(p) ? nothing : deepcopy(opt))
    end
    push!(opts,t_opts)
    if length(ps) > 1
        for p in ps[2:end]
            push!(opts,isnothing(p) ? nothing : deepcopy(opt))
        end
    end
    return opts
end


KernelFunctions.kernelmatrix(k::KernelWrapper,X) = kernelmatrix(k.kernel,X,obsdim=1)
KernelFunctions.kernelmatrix(k::KernelSumWrapper,X) = sum(w*kernelmatrix(kernel,X) for (w,kernel) in zip(k.weights,k.kernelwrappers))
KernelFunctions.kernelmatrix(k::KernelProductWrapper,X) = reduce(hadamard,kernelmatrix(kernel,X) for kernel in k.kernelwrappers)
KernelFunctions.kernelmatrix(k::AbstractKernelWrapper,X,Y) = kernelmatrix(k.kernel,X,Y,obsdim=1)
KernelFunctions.kernelmatrix(k::KernelSumWrapper,X,Y) = sum(w*kernelmatrix(kernel,X,Y) for (w,kernel) in zip(k.weights,k.kernelwrappers))
KernelFunctions.kernelmatrix(k::KernelProductWrapper,X,Y) = reduce(hadamard,kernelmatrix(kernel,X,Y) for kernel in k.kernelwrappers)
KernelFunctions.kerneldiagmatrix(k::KernelWrapper,X) = kerneldiagmatrix(k.kernel,X,obsdim=1)
KernelFunctions.kerneldiagmatrix(k::KernelSumWrapper,X) = sum(w*kerneldiagmatrix(kernel,X) for (w,kernel) in zip(k.weights,k.kernelwrappers))
KernelFunctions.kerneldiagmatrix(k::KernelProductWrapper,X) = reduce(hadamard,kerneldiagmatrix(kernel,X) for kernel in k.kernelwrappers)
