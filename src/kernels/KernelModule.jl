"""
     Module for the kernel functions, also create kernel matrices
     Mostly from the list http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
     Kernels are created by calling the constructor (available right now :
     RBFKernel,
     Arguments are kernel specific, see for each one
"""
#LaplaceKernel, SigmoidKernel, ARDKernel, PolynomialKernel, Matern3_2Kernel, Matern5_2Kernel
module KernelModule

using LinearAlgebra
using Distances
using SpecialFunctions
include("hyperparameters/HyperParametersModule.jl")
using .HyperParametersModule:
    Bound,
        OpenBound,
        ClosedBound,
        NullBound,
    Interval,
        interval,
    HyperParameter,
    HyperParameters,
        getvalue,
        setvalue!,
        checkvalue,
        gettheta,
        checktheta,
        settheta!,
        lowerboundtheta,
        upperboundtheta,
        update!,
        setfixed!,
        setfree!

import Base: *, +, getindex, show
export Kernel, KernelSum, KernelProduct
export RBFKernel, SEKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel, MaternKernel, Matern5_2Kernel
export kernelmatrix,kernelmatrix!,kerneldiagmatrix,kerneldiagmatrix!
export computeIndPointsJ
export apply_gradients_lengthscale!, apply_gradients_variance!, apply_gradients!
export kernelderivativematrix_K
export kernelderivativematrix,kernelderivativediagmatrix
export InnerProduct, SquaredEuclidean, Identity
export compute_hyperparameter_gradient
export compute,plotkernel
export getvalue,setvalue!,setfixed!,setfree!
export getlengthscales, getvariance
export isARD,isIso
export HyperParameter,HyperParameters,Interval, OpenBound,  NullBound
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

isARD(k::Kernel{T,KT}) where {T<:Real,KT<:KernelType} = KT <: ARDKernel
isIso(k::Kernel{T,KT}) where {T<:Real,KT<:KernelType} = KT <: IsoKernel

function Base.show(io::IO,k::Kernel{T,KT}) where {T,KT}
    print(io,"$(k.fields.name)"*(isARD(k) ? " ARD" : "")*" kernel, with variance $(getvariance(k)) and lengthscales $(getlengthscales(k))")
end

include("KernelSum.jl")
include("KernelProduct.jl")
include("RBF.jl")
include("Matern.jl")
include("KernelMatrix.jl")
include("KernelGradients.jl")

"""Standard conversion when giving scalar and not vectors"""
function compute(k::Kernel{T,KT},X1::T,X2::T) where {T<:Real,KT<:KernelType}
    compute(k,[X1],[X2])
end

"""Function to determine most adapted type between a selection"""
function floattype(T_i::DataType...)
    T_max = promote_type(T_i...)
    T_max <: AbstractFloat ? T_max : Float64
end
#
# """
#     Laplace Kernel
# """
# mutable struct LaplaceKernel{T,KT<:KernelType} <: Kernel{T,KT}
#     @kernelfunctionfields
#     function LaplaceKernel{T,KT}(θ::T=1.0;variance::T=1.0,dim::Integer=0,ARD::Bool=false) where {T<:AbstractFloat,KT<:KernelType}
#         return new("Laplace",
#         HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
#         HyperParameters{T}([θ],[interval(OpenBound(zero(T)),nothing)]),
#         1,SquaredEuclidean)
#     end
# end
#
# function LaplaceKernel(θ::T1=1.0;variance::T2=one(T1)) where {T1<:Real,T2<:Real}
#      LaplaceKernel{floattype(T1,T2)}(θ,variance=variance)
#  end
#
# "Apply kernel functions on vector"
# function compute(k::LaplaceKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     if X1 == X2
#       return (variance ? getvalue(k.variance) : 1.0)
#     end
#     return (variance ? getvalue(k.variance) : 1.0)*exp(-k.distance(X1,X2)/(k.param[1]))
# end
# #
# "Compute kernel gradients given the vectors"
# function compute_deriv(k::LaplaceKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     a = k.distance(X1,X2)
#     if a != 0
#         grad = a/((k.param[1])^2)*compute(k,X1,X2)
#         if variance
#             return [getvalue(k.variance)*grad,compute(k,X1,X2)]
#         else
#             return [grad]
#         end
#     else
#       return [0.0]
#     end
# end
#
# #TODO Not correct
# function compute_point_deriv(k::LaplaceKernel{T},X1::Vector{T},X2::Vector{T}) where T
#     if X1 == X2
#         return zeros(X1)
#     else
#         return getvalue(k.variance)*(-(X1-X2))./(k.param[1]^2).*compute(k,X1,X2)
#     end
# end
#
# """
#     Sigmoid Kernel
#     tanh(p1*d+p2)
# """
# mutable struct SigmoidKernel{T,KT<:KernelType} <: Kernel{T,KT}
#     @kernelfunctionfields
#     function SigmoidKernel{T,KT}(θ::Vector{T}=[1.0,0.0],variance::Float64=1.0,dim::Integer=0,ARD::Bool=false) where {T<:Real,KT<:KernelType}
#         return new("Sigmoid",
#         HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
#         HyperParameters{T}(θ,[interval(NullBound{T}(),NullBound{T}()), interval(NullBound{T}(),NullBound{T}())]),
#         length(θ),InnerProduct)
#     end
# end
# function SigmoidKernel(θ::Vector{T1}=[1.0,0.0];variance::T2=one(T1)) where {T1<:Real,T2<:Real}
#      SigmoidKernel{floattype(T1,T2)}(θ,variance=variance)
#  end
#
# "Apply kernel functions on vector"
# function compute(k::SigmoidKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     return (variance ? getvalue(k.variance) : 1.0)*tanh(k.param[1]*k.distance(X1,X2)+k.param[2])
# end
# #
# "Compute kernel gradients given the vectors"
# function compute_deriv(k::SigmoidKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#
#     grad_1 = k.distance(X1,X2)*(1-compute(k,X1,X2)^2)
#     grad_2 = (1-compute(k,X1,X2)^2)
#     if variance
#         return [getvalue(k.variance)*grad_1,getvalue(k.variance)*grad_2,compute(k,X1,X2)]
#     else
#         return [grad_1,grad_2]
#     end
# end
#
# function compute_point_deriv(k::SigmoidKernel{T},X1::Vector{T},X2::Vector{T}) where T
#     return k.param[1]*X2.*(1-compute(k,X1,X2)^2)
# end
#
# """
#     Polynomial Kernel
#     (p1*d+p2)^p3
# """
# mutable struct PolynomialKernel{T,KT<:KernelType} <: Kernel{T,KT}
#     @kernelfunctionfields
#     function PolynomialKernel{T,KT}(θ::Vector{T}=[1.0,0.0,2.0];variance::T=1.0) where {T<:Real,KT<:KernelType}
#         return new("Polynomial",HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
#                                 HyperParameters{T}(θ,[interval(NullBound{T}(),NullBound{T}()) for i in 1:length(θ)]),
#                                 length(θ),InnerProduct)
#     end
# end
# function PolynomialKernel(θ::Vector{T1}=[1.0,0.0,2.0];variance::T2=one(T1)) where {T1<:Real,T2<:Real}
#      PolynomialKernel{floattype(T1,T2)}(θ,variance=variance)
#  end
#
# "Apply kernel functions on vector"
# function compute(k::PolynomialKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     return (variance ? getvalue(k.variance) : 1.0)*(k.param[1]*k.distance(X1,X2)+k.param[2])^getvalue(k.param[3])
# end
# #
# "Compute kernel gradients given the vectors"
# function compute_deriv(k::PolynomialKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     grad_1 = k.param[3]*k.distance(X1,X2)*(k.param[1]*k.distance(X1,X2)+k.param[2])^(k.param[3]-1)
#     grad_2 = k.param[3]*(k.param[1]*k.distance(X1,X2)+k.param[2])^(k.param[3]-1)
#     grad_3 = log(k.param[1]*k.distance(X1,X2)+k.param[2])*compute(k,X1,X2)
#     if variance
#         return [getvalue(k.variance)*grad_1,getvalue(k.variance)*grad_2,getvalue(k.variance)*grad_3,compute(k,X1,X2)]
#     else
#         return [grad_1,grad_2,grad_3]
#     end
# end
#
# function compute_point_deriv(k::PolynomialKernel{T},X1::Vector{T},X2::Vector{T}) where T
#     return k.param[3]*k.param[1]*X2*(k.param[1]*k.distance(X1,X2)+k.param[2])^(k.param[3]-1)
# end
###################################################### OLD STUFF
#
# """
#     ARD Kernel
# """
# mutable struct ARDKernel{T} <: Kernel{T}
#     @kernelfunctionfields
#     function ARDKernel{T}(θ::Vector{T}=[1.0];dim::Int64=0,variance::T=1.0) where {T<:Real}
#         if length(θ)==1 && dim ==0
#             error("You defined an ARD kernel without precising the number of dimensions
#                              Please set dim in your kernel initialization or use ARDKernel(X,θ)")
#         elseif dim!=0 && (length(θ)!=dim && length(θ)!=1)
#             @warn "You did not use the same dimension for your params and dim, using the first value of params for all dimensions"
#             θ = ones(dim,T=T)*θ[1]
#         elseif length(θ)==1 && dim!=0
#             θ = ones(dim)*θ[1]
#         end
#         intervals = [interval(OpenBound{T}(zero(T)),NullBound{T}()) for i in 1:length(θ)]
#         return new("ARD",HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
#                         HyperParameters{T}(θ,intervals),
#                         length(θ),SquaredEuclidean)
#     end
# end
#
# function ARDKernel(θ::Vector{T1}=[1.0];dim::Int64=0,variance::T2=one(T1)) where {T1<:Real,T2<:Real}
#      ARDKernel{floattype(T1,T2)}(θ,dim=dim,variance=variance)
# end
# # function ARDKernel(θ::T1=1.0;dim::Int64=0,variance::T2=one(T1)) where {T1<:Real,T2<:Real}
# #      ARDKernel{floattype(T1,T2)}([θ],dim=dim,variance=variance)
# # end
#
# "Apply kernel functions on vector"
# function compute(k::ARDKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     if X1==X2
#         return (variance ? getvalue(k.variance) : 1.0)
#     end
#     # return (variance ? getvalue(k.variance) : 1.0)*exp(-0.5*sum(((X1-X2)./getvalue(k.param)).^2))
#     return (variance ? getvalue(k.variance) : 1.0)*exp(-0.5*sum(((X1[i]-X2[i])/getvalue(k.param[i]))^2 for i in 1:length(X1)))
# end
# #
# "Compute kernel gradients given the vectors"
# function compute_deriv(k::ARDKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     D = length(X1)
#     if variance
#         if X1 == X2
#             g = zeros(k.Nparam+1)
#             g[end] = compute(k,X1,X2,false)
#             return g
#         else
#             return getvalue(k.variance).*[i<(D+1) ? (X1[i]-X2[i])^2/(getvalue(k.param[i]))^3 : 1.0/getvalue(k.variance) for i in 1:(D+1)]#.*compute(k,X1,X2,false)
#         end
#     else
#         if X1 == X2
#             return zeros(k.Nparam)
#         else
#             return [(X1[i]-X2[i])^2/(getvalue(k.param[i]))^3 for i in 1:length(X1)]#.*compute(k,X1,X2,false)
#         end
#     end
# end
#
# function compute_point_deriv(k::ARDKernel{T},X1::Vector{T},X2::Vector{T}) where T
#     if X1==X2
#         return zeros(X1)
#     end
#     return -2*[((X1[i]-X2[i])/getvalue(k.param[i]))^2 for i in 1:length(X1)]#.*compute(k,X1,X2)
# end
#
# """
#     Matern 3/2 Kernel
#     d= ||X1-X2||^2
#     (1+\frac{√(3)d}{ρ})exp(-\frac{√(3)d}{ρ})
# """
# mutable struct Matern3_2Kernel{T,KT<:KernelType} <: Kernel{T,KT}
#     @kernelfunctionfields
#     function Matern3_2Kernel{T,KT}(θ::Float64=1.0;variance::Float64=1.0) where {T<:Real,KT<:KernelType}
#         return new("Matern3_2Kernel",HyperParameter(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
#                                     HyperParameters([θ],[interval(OpenBound{T}(zero(T)),NullBound{T}())]),
#                                     length(θ),SquaredEuclidean)
#     end
# end
# function Matern3_2Kernel(θ::T1=1.0;variance::T2=one(T1)) where {T1<:Real,T2<:Real}
#      Matern3_2Kernel{floattype(T1,T2)}(θ,variance=variance)
#  end
# "Apply kernel functions on vector"
# function compute(k::Matern3_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     d = sqrt(3.0)*k.distance(X1,X2)
#     return (variance ? getvalue(k.variance) : 1.0)*(1.0+d/k.param[1])*exp(-d/k.param[1])
# end
# #
# "Compute kernel gradients given the vectors"
# function compute_deriv(k::Matern3_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     d = sqrt(3.0)*k.distance(X1,X2)
#     grad_1 = -d*(1+d/k.param[1]+1/(k.param[1])^2)*exp(-d/k.param[1])
#     if variance
#         return [getvalue(k.variance)*grad_1,compute(k,X1,X2)]
#     else
#         return [grad_1]
#     end
# end
#
# function compute_point_deriv(k::Matern3_2Kernel{T},X1::Vector{T},X2::Vector{T}) where T
#     ### TODO
# end
#
# """
#     Matern 5/2 Kernel
#     d= ||X1-X2||^2
#     (1+\frac{√(5)d}{ρ}+\frac{5d^2}{3ρ^2})exp(-\frac{-√(5)d}{ρ})
# """
# mutable struct Matern5_2Kernel{T,KT<:KernelType} <: Kernel{T,KT}
#     @kernelfunctionfields
#     function Matern5_2Kernel{T,KT}(θ::T=1.0;variance::T=1.0) where {T<:Real,KT<:KernelType}
#         return new("Matern5_2Kernel",HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}())),
#                                     HyperParameters{T}([θ],[interval(OpenBound{T}(zero(T)),NullBound{T}())]),
#                                     length(θ),SquaredEuclidean)
#     end
# end
#
# function Matern5_2Kernel(θ::T1=1.0;variance::T2=one(T1)) where {T1<:Real,T2<:Real}
#      Matern5_2Kernel{floattype(T1,T2)}(θ,variance=variance)
#  end
#
# "Apply kernel functions on vector"
# function compute(k::Matern5_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     d = sqrt(5.0)*k.distance(X1,X2)
#     return (variance ? getvalue(k.variance) : 1.0)*(1.0+d/k.param[1]+d^2/(3.0*k.param[1]^2))*exp(-d/k.param[1])
# end
# #
# "Compute kernel gradients given the vectors"
# function compute_deriv(k::Matern5_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
#     d = sqrt(5.0)*k.distance(X1,X2)
#     grad_1 = -d*(1+d^2/k.param[1]+(3*d+d^3)/(3*k.param[1]^2)+2*d^2/(3*k.param[1]^3))*exp(-d/k.param[1])
#     if variance
#         return [getvalue(k.variance)*grad_1,compute(k,X1,X2)]
#     else
#         return [grad_1]
#     end
# end
#
# function compute_point_deriv(k::Matern5_2Kernel{T},X1::Vector{T},X2::Vector{T}) where T
#     ### TODO
# end

# type Kernel
#     name::String #Type of function
#     kernel_function::Function # Kernel function
#     coeff::Float64 # for the kernel
#     derivative_kernel::Function # Derivative of the kernel function (used for hyperparameter optimization)
#     derivative_point::Function # Derivative of the kernel function given the feature vector (returns a gradient)
#     param::Float64 #Hyperparameters for the kernel function (depends of the function)
#     Nparams::Int64 #Number of hyperparameters
#     compute::Function #General computational function
#     compute_deriv::Function #General derivative function
#     compute_point_deriv::Function #Derivative function for inducing points
#     #Constructor
#     function Kernel(kernel, coeff::Float64; dim=0, params=[1.0])
#         this = new(kernel)
#         this.coeff = coeff
#         this.Nparams = 1
#         elseif kernel=="ARD"
#             this.kernel_function = ARD
#             this.derivative_kernel = deriv_ARD
#             this.derivative_point = deriv_point_ARD
#             if length(params)==1 && dim==0
#                 error("You defined an ARD kernel without precising the number of dimensions
#                 Please set dim in your kernel initialization")
#             elseif dim!=0 && (length(params)!=dim && length(params)!=1)
#                 @warn "You did not use the same dimension for your params and dim, using the first value of params for all dimensions"
#                 this.param = ones(dim)*params[1]
#             elseif length(params)==1 && dim!=0
#                 this.param = ones(dim)*params[1]
#             else
#                 this.param = params
#             end
#         elseif kernel=="linear"
#             this.kernel_function = linear
#             this.derivative_kernel = deriv_linear
#             this.derivative_point = deriv_point_linear
#             this.param = 0
#         elseif kernel=="laplace"
#             this.kernel_function = laplace
#             this.derivative_kernel = deriv_laplace
#             this.derivative_point = 0 # TODO
#             this.param = params
#         elseif kernel=="abel"
#             this.kernel_function = abel
#             this.derivative_kernel = deriv_abel
#             this.derivative_point = 0 # TODO
#             this.param = params
#         elseif kernel=="imk"
#             this.kernel_function = imk
#             this.derivative_kernel = deriv_imk
#             this.derivative_point = 0 # TODO
#             this.param = params
#         else
#             error("Kernel function $(kernel) not available, options are : rbf, ARD, quadra, linear, laplace, abel, imk")
#         end
#         this.compute = function(X1,X2)
#                 this.kernel_function(X1,X2,this.param)
#             end
#
#         this.compute_point_deriv = function(X1,X2)
#                 this.derivative_point(X1,X2,this.param)
#             end
#         return this
#     end
# end



#The ANOVA kernel is not super well defined for now #TODO
# function anov(x,i,y,j,k,K)
#     if (i == 1 || j == 1)
#         a = 0;
#     else
#         # Retrieve the value from the cache
#         a = K[i-1, j-1, k];
#     end
# # Compute a linear kernel
#     lin_k = x[i] * y[j];
#     if (k == 1)
#         return a + lin_k;
#     end
#     if (i == 1 || j == 1)
#         return a;
#     end
# # Retrieve the value from the cache
#     return a + lin_k * K[i-1, j-1, k-1];
# end
# #ANOVA Kernel with parameters n,sigma and d
# function ANOVA(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
#     #Using dynamic programming
#     K = zeros(n,n,p)
#     for k in 1:p
#         for i in 1:n
#             for j in 1:n
#                 K[i, j, k] = anov(X1, i, X2, j, k, K);
#             end
#         end
#     end
# # Get the final result
#     return K[n, n, p];
# end


#Abel Kernel
function abel(X1::Vector{Float64},X2::Array{Float64,1},Θ)
  if X1==X2
    return 1
  end
  exp(-Θ[1]*norm(X1-X2,1))
end

function deriv_abel(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  a = norm(X1-X2,1)
  -a*exp(-Θ*a)
end

#Inverse Multiquadratic Kernel
function imk(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  1/(sqrt(norm(X1-X2)^2+Θ[1]^2))
end

function deriv_imk(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  -θ[1]/((sqrt(norm(X1-X2)+Θ[1]))^3)
end

#Linear Kernel
function linear(X1::Array{Float64,1},X2::Array{Float64,1},θ)
    dot(X1,X2)
end

function deriv_linear(X1::Array{Float64,1},X2::Array{Float64,1},θ)
    return 0
end

function deriv_point_linear(X1::Array{Float64,1},X2::Array{Float64,1},θ)
    X2
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

function apply_gradients!(kernel::KernelSum,gradients,variance::Bool=true)
    for i in 1:kernel.Nkernels
        apply_gradients!(kernel.kernel_array[i],gradients[i],true)
    end
end

function apply_gradients!(kernel::KernelProduct,gradients,variance::Bool=true)
    for i in 1:kernel.Nkernels
        apply_gradients!(kernel.kernel_array[i],gradients[i],false);
    end
    if variance
        update!(kernel.variance,gradients[end][1]);
    end
end

end #end of module
