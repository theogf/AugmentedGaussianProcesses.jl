# """

# """
"""
     Module for the kernel functions, also create kernel matrices
     Mostly from the list http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
     Kernels are created by calling the constructor (available right now :
     RBFKernel, LaplaceKernel, SigmoidKernel, ARDKernel, PolynomialKernel, Matern3_2Kernel, Matern5_2Kernel
     Arguments are kernel specific, see for each one
"""
module KernelFunctions

using LinearAlgebra
include("HyperParameters/HyperParametersMod.jl")
using .HyperParametersMod:
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


"Macro giving common fields for all kernels"
macro kernelfunctionfields()
    return esc(quote
        name::String #Name of the kernel function
        variance::HyperParameter{T} #variance of the kernel
        param::HyperParameters{T} #Parameters of the kernel
        Nparam::Int64 #Number of parameters
        distance::Function
    end)
end

import Base: *, +, getindex
export Kernel, KernelSum, KernelProduct
export RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel, Matern3_2Kernel, Matern5_2Kernel
export kernelmatrix,kernelmatrix!,diagkernelmatrix,diagkernelmatrix!,computeIndPointsJ
export derivativekernelmatrix,derivativediagkernelmatrix,compute_hyperparameter_gradient,apply_gradients!
export InnerProduct, SquaredEuclidean, Identity
export compute,plotkernel
export getvalue,setvalue!,setfixed!,setfree!


"Abstract type for all kernels"
abstract type Kernel{T<:AbstractFloat} end;

"Standard conversion when giving scalar and not vectors"
function compute(k::Kernel,X1::T,X2::T) where {T<:Real}
    compute(k,[X1],[X2])
end

"Return the inner product between two vectors"
InnerProduct(X1,X2) = dot(X1,X2);
"Return the squared euclidian distance between to vectors"
SquaredEuclidean(X1,X2) = norm(X1-X2,2);
"Return the same set of arguments"
Identity(X1,X2) = (X1,X2);

"Structure to combine kernels together by addition, can be created by using the constructor with an array or simply using Base.+"
mutable struct KernelSum{T<:AbstractFloat} <: Kernel{T}
    @kernelfunctionfields()
    kernel_array::Vector{Kernel{T}} #Array of summed kernels
    Nkernels::Int64 #Number of kernels
    "Inner KernelSum constructor taking an array of kernels"
    function KernelSum{T}(kernels::AbstractArray) where {T<:AbstractFloat}
        this = new("Sum of kernels")
        this.kernel_array = deepcopy(Vector{Kernel{T}}(kernels))
        this.Nkernels = length(this.kernel_array)
        this.distance = Identity
        return this
    end
end

"Apply kernel functions on vector"
function compute(k::KernelSum{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where {T}
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
"Structure to combine kernels together by addition, can be created by using the constructor with an array or simply using Base.*"
mutable struct KernelProduct{T<:AbstractFloat} <: Kernel{T}
    @kernelfunctionfields
    kernel_array::Vector{Kernel{T}} #Array of multiplied kernels
    Nkernels::Int64 #Number of multiplied kernels
    "Inner KernelProduct constructor taking an array of kernels"
    function KernelProduct{T}(kernels::Vector{Kernel{T}}) where {T<:AbstractFloat}
        this = new("Product of kernels",
                HyperParameter{T}(1.0,interval(OpenBound(zero(T)),nothing),fixed=false))
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
function floattype(T_i::DataType...)
    T_max = promote_type(T_i...)
    T_max <: AbstractFloat ? T_max : Float64
end

"""
    Gaussian (RBF) Kernel
"""
mutable struct RBFKernel{T<:AbstractFloat} <: Kernel{T}
    @kernelfunctionfields
    function RBFKernel{T}(θ::T=1.0;variance::T=1.0) where {T<:AbstractFloat}
        return new("RBF",
        HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
        HyperParameters([θ],[interval(OpenBound(zero(T)),NullBound{T}())]),
        1,SquaredEuclidean)
    end
end
function RBFKernel(θ::T1=1.0;variance::T2=one(T1)) where {T1<:Real,T2<:Real}
     RBFKernel{floattype(T1,T2)}(θ,variance=variance)
 end
"Apply kernel functions on vector"
function compute(k::RBFKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    if X1 == X2
      return (variance ? getvalue(k.variance) : 1.0)
    end
    @assert k.distance(X1,X2) > 0  "Problem with distance computation"
    return (variance ? getvalue(k.variance) : 1.0)*exp(-0.5*(k.distance(X1,X2)/getvalue(k.param[1]))^2)
end
#
"Compute kernel gradients given the vectors"
function compute_deriv(k::RBFKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    a = k.distance(X1,X2)
    grad = a^2/(getvalue(k.param[1])^3)*compute(k,X1,X2,false)
    if variance
        return [getvalue(k.variance)*grad,compute(k,X1,X2)]
    else
        return [grad]
    end
end

#TODO probably not right
function compute_point_deriv(k::RBFKernel{T},X1::Vector{T},X2::Vector{T}) where T
    if X1 == X2
        return zeros(X1)
    else
        return getvalue(k.variance)*(-(X1-X2))./((k.param[1])^2).*compute(k,X1,X2)
    end
end

"""
    Laplace Kernel
"""
mutable struct LaplaceKernel{T} <: Kernel{T}
    @kernelfunctionfields
    function LaplaceKernel{T}(θ::T=1.0;variance::T=1.0) where {T<:AbstractFloat}
        return new("Laplace",
        HyperParameter{T}(variance,interval(OpenBound(zero(T)),nothing),fixed=false),
        HyperParameters{T}([θ],[interval(OpenBound(zero(T)),nothing)]),
        1,SquaredEuclidean)
    end
end

function LaplaceKernel(θ::T1=1.0;variance::T2=one(T1)) where {T1<:Real,T2<:Real}
     LaplaceKernel{floattype(T1,T2)}(θ,variance=variance)
 end

"Apply kernel functions on vector"
function compute(k::LaplaceKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    if X1 == X2
      return (variance ? getvalue(k.variance) : 1.0)
    end
    return (variance ? getvalue(k.variance) : 1.0)*exp(-k.distance(X1,X2)/(k.param[1]))
end
#
"Compute kernel gradients given the vectors"
function compute_deriv(k::LaplaceKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    a = k.distance(X1,X2)
    if a != 0
        grad = a/((k.param[1])^2)*compute(k,X1,X2)
        if variance
            return [getvalue(k.variance)*grad,compute(k,X1,X2)]
        else
            return [grad]
        end
    else
      return [0.0]
    end
end

#TODO Not correct
function compute_point_deriv(k::LaplaceKernel{T},X1::Vector{T},X2::Vector{T}) where T
    if X1 == X2
        return zeros(X1)
    else
        return getvalue(k.variance)*(-(X1-X2))./(k.param[1]^2).*compute(k,X1,X2)
    end
end

"""
    Sigmoid Kernel
    tanh(p1*d+p2)
"""
mutable struct SigmoidKernel{T} <: Kernel{T}
    @kernelfunctionfields
    function SigmoidKernel{T}(θ::Vector{T}=[1.0,0.0];variance::Float64=1.0) where {T<:Real}
        return new("Sigmoid",
        HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
        HyperParameters{T}(θ,[interval(NullBound{T}(),NullBound{T}()), interval(NullBound{T}(),NullBound{T}())]),
        length(θ),InnerProduct)
    end
end
function SigmoidKernel(θ::Vector{T1}=[1.0,0.0];variance::T2=one(T1)) where {T1<:Real,T2<:Real}
     SigmoidKernel{floattype(T1,T2)}(θ,variance=variance)
 end

"Apply kernel functions on vector"
function compute(k::SigmoidKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    return (variance ? getvalue(k.variance) : 1.0)*tanh(k.param[1]*k.distance(X1,X2)+k.param[2])
end
#
"Compute kernel gradients given the vectors"
function compute_deriv(k::SigmoidKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T

    grad_1 = k.distance(X1,X2)*(1-compute(k,X1,X2)^2)
    grad_2 = (1-compute(k,X1,X2)^2)
    if variance
        return [getvalue(k.variance)*grad_1,getvalue(k.variance)*grad_2,compute(k,X1,X2)]
    else
        return [grad_1,grad_2]
    end
end

function compute_point_deriv(k::SigmoidKernel{T},X1::Vector{T},X2::Vector{T}) where T
    return k.param[1]*X2.*(1-compute(k,X1,X2)^2)
end

"""
    Polynomial Kernel
    (p1*d+p2)^p3
"""
mutable struct PolynomialKernel{T} <: Kernel{T}
    @kernelfunctionfields
    function PolynomialKernel{T}(θ::Vector{T}=[1.0,0.0,2.0];variance::T=1.0) where {T<:Real}
        return new("Polynomial",HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
                                HyperParameters{T}(θ,[interval(NullBound{T}(),NullBound{T}()) for i in 1:length(θ)]),
                                length(θ),InnerProduct)
    end
end
function PolynomialKernel(θ::Vector{T1}=[1.0,0.0,2.0];variance::T2=one(T1)) where {T1<:Real,T2<:Real}
     PolynomialKernel{floattype(T1,T2)}(θ,variance=variance)
 end

"Apply kernel functions on vector"
function compute(k::PolynomialKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    return (variance ? getvalue(k.variance) : 1.0)*(k.param[1]*k.distance(X1,X2)+k.param[2])^getvalue(k.param[3])
end
#
"Compute kernel gradients given the vectors"
function compute_deriv(k::PolynomialKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    grad_1 = k.param[3]*k.distance(X1,X2)*(k.param[1]*k.distance(X1,X2)+k.param[2])^(k.param[3]-1)
    grad_2 = k.param[3]*(k.param[1]*k.distance(X1,X2)+k.param[2])^(k.param[3]-1)
    grad_3 = log(k.param[1]*k.distance(X1,X2)+k.param[2])*compute(k,X1,X2)
    if variance
        return [getvalue(k.variance)*grad_1,getvalue(k.variance)*grad_2,getvalue(k.variance)*grad_3,compute(k,X1,X2)]
    else
        return [grad_1,grad_2,grad_3]
    end
end

function compute_point_deriv(k::PolynomialKernel{T},X1::Vector{T},X2::Vector{T}) where T
    return k.param[3]*k.param[1]*X2*(k.param[1]*k.distance(X1,X2)+k.param[2])^(k.param[3]-1)
end

"""
    ARD Kernel
"""
mutable struct ARDKernel{T} <: Kernel{T}
    @kernelfunctionfields
    function ARDKernel{T}(θ::Vector{T}=[1.0];dim::Int64=0,variance::T=1.0) where {T<:Real}
        if length(θ)==1 && dim ==0
            error("You defined an ARD kernel without precising the number of dimensions
                             Please set dim in your kernel initialization or use ARDKernel(X,θ)")
        elseif dim!=0 && (length(θ)!=dim && length(θ)!=1)
            @warn "You did not use the same dimension for your params and dim, using the first value of params for all dimensions"
            θ = ones(dim,T=T)*θ[1]
        elseif length(θ)==1 && dim!=0
            θ = ones(dim)*θ[1]
        end
        intervals = [interval(OpenBound{T}(zero(T)),NullBound{T}()) for i in 1:length(θ)]
        return new("ARD",HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
                        HyperParameters{T}(θ,intervals),
                        length(θ),SquaredEuclidean)
    end
end

function ARDKernel(θ::Vector{T1}=[1.0];dim::Int64=0,variance::T2=one(T1)) where {T1<:Real,T2<:Real}
     ARDKernel{floattype(T1,T2)}(θ,dim=dim,variance=variance)
end
# function ARDKernel(θ::T1=1.0;dim::Int64=0,variance::T2=one(T1)) where {T1<:Real,T2<:Real}
#      ARDKernel{floattype(T1,T2)}([θ],dim=dim,variance=variance)
# end

"Apply kernel functions on vector"
function compute(k::ARDKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    if X1==X2
        return (variance ? getvalue(k.variance) : 1.0)
    end
    return (variance ? getvalue(k.variance) : 1.0)*exp(-0.5*sum(((X1-X2)./k.param.hyperparameters).^2))
end
#
"Compute kernel gradients given the vectors"
function compute_deriv(k::ARDKernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    if X1 == X2
        grad = zeros(k.Nparam)
    else
        grad = (X1-X2).^2 ./(getvalue(k.param).^3).*compute(k,X1,X2,false)
    end
    if variance
        return vcat(getvalue(k.variance)*grad,compute(k,X1,X2,false))
    else
        return grad
    end
end

function compute_point_deriv(k::ARDKernel{T},X1::Vector{T},X2::Vector{T}) where T
    if X1==X2
        return zeros(X1)
    end
    return -2*(X1-X2)./(k.param.hyperparameters.^2).*compute(k,X1,X2)
end

"""
    Matern 3/2 Kernel
    d= ||X1-X2||^2
    (1+\frac{√(3)d}{ρ})exp(-\frac{√(3)d}{ρ})
"""
mutable struct Matern3_2Kernel{T} <: Kernel{T}
    @kernelfunctionfields
    function Matern3_2Kernel{T}(θ::Float64=1.0;variance::Float64=1.0) where {T<:Real}
        return new("Matern3_2Kernel",HyperParameter(variance,interval(OpenBound{T}(zero(T)),NullBound{T}()),fixed=false),
                                    HyperParameters([θ],[interval(OpenBound{T}(zero(T)),NullBound{T}())]),
                                    length(θ),SquaredEuclidean)
    end
end
function Matern3_2Kernel(θ::T1=1.0;variance::T2=one(T1)) where {T1<:Real,T2<:Real}
     Matern3_2Kernel{floattype(T1,T2)}(θ,variance=variance)
 end
"Apply kernel functions on vector"
function compute(k::Matern3_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    d = sqrt(3.0)*k.distance(X1,X2)
    return (variance ? getvalue(k.variance) : 1.0)*(1.0+d/k.param[1])*exp(-d/k.param[1])
end
#
"Compute kernel gradients given the vectors"
function compute_deriv(k::Matern3_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    d = sqrt(3.0)*k.distance(X1,X2)
    grad_1 = -d*(1+d/k.param[1]+1/(k.param[1])^2)*exp(-d/k.param[1])
    if variance
        return [getvalue(k.variance)*grad_1,compute(k,X1,X2)]
    else
        return [grad_1]
    end
end

function compute_point_deriv(k::Matern3_2Kernel{T},X1::Vector{T},X2::Vector{T}) where T
    ### TODO
end

"""
    Matern 5/2 Kernel
    d= ||X1-X2||^2
    (1+\frac{√(5)d}{ρ}+\frac{5d^2}{3ρ^2})exp(-\frac{-√(5)d}{ρ})
"""
mutable struct Matern5_2Kernel{T} <: Kernel{T}
    @kernelfunctionfields
    function Matern5_2Kernel{T}(θ::T=1.0;variance::T=1.0) where {T<:Real}
        return new("Matern5_2Kernel",HyperParameter{T}(variance,interval(OpenBound{T}(zero(T)),NullBound{T}())),
                                    HyperParameters{T}([θ],[interval(OpenBound{T}(zero(T)),NullBound{T}())]),
                                    length(θ),SquaredEuclidean)
    end
end

function Matern5_2Kernel(θ::T1=1.0;variance::T2=one(T1)) where {T1<:Real,T2<:Real}
     Matern5_2Kernel{floattype(T1,T2)}(θ,variance=variance)
 end

"Apply kernel functions on vector"
function compute(k::Matern5_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    d = sqrt(5.0)*k.distance(X1,X2)
    return (variance ? getvalue(k.variance) : 1.0)*(1.0+d/k.param[1]+d^2/(3.0*k.param[1]^2))*exp(-d/k.param[1])
end
#
"Compute kernel gradients given the vectors"
function compute_deriv(k::Matern5_2Kernel{T},X1::Vector{T},X2::Vector{T},variance::Bool=true) where T
    d = sqrt(5.0)*k.distance(X1,X2)
    grad_1 = -d*(1+d^2/k.param[1]+(3*d+d^3)/(3*k.param[1]^2)+2*d^2/(3*k.param[1]^3))*exp(-d/k.param[1])
    if variance
        return [getvalue(k.variance)*grad_1,compute(k,X1,X2)]
    else
        return [grad_1]
    end
end

function compute_point_deriv(k::Matern5_2Kernel{T},X1::Vector{T},X2::Vector{T}) where T
    ### TODO
end

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

function apply_gradients!(kernel::Kernel,gradients,variance::Bool=true)
    update!(kernel.param,gradients[kernel.Nparam ==1 ? 1 : 1:kernel.Nparam])
    update!(kernel.param,gradients[1:kernel.Nparam])
    if variance
        update!(kernel.variance,gradients[end])
    end
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

include("KernelMatrix.jl")

end #end of module
