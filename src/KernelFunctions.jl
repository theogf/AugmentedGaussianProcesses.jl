# """
#     Module for the kernel functions, also create kernel matrices
#         From the list http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
# """

module KernelFunctions

#Simple tool to define macros
macro kernelfunctionfields()
    return esc(quote
        name::String #Name of the kernel function
        weight::HyperParameter #Weight of the kernel
        hyperparameters::AbstractHyperParameter #Parameters of the kernel
        Nparameters::Int64 #Number of parameters
        pairwisefunction::Function
    end)
end

using GradDescent
using Plots;
pyplot();



import Base: *, +, getindex
export Kernel, KernelSum, KernelProduct
export RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel
export kernelmatrix,kernelmatrix!,diagkernelmatrix,diagkernelmatrix!
export derivativekernelmatrix,derivativediagkernelmatrix,compute_hyperparameter_gradient,apply_gradients!
export compute,plotkernel



abstract type Kernel end;

abstract type AbstractHyperParameter end;

mutable struct HyperParameter <: AbstractHyperParameter
    value::Float64
    fixed::Bool
    opt::Optimizer
    function HyperParameter(θ::Float64;opt::Optimizer=Adam())
        new(θ,false,opt)
    end
end

mutable struct HyperParameters <: AbstractHyperParameter
    hyperparameters::Array{HyperParameter,1}
    function HyperParameters(θ::Array{Float64})
        this = new(Array{HyperParameter,1}())
        for x in θ
            push!(this.hyperparameters,HyperParameter(x))
        end
        return this
    end
end


getvalue(param::HyperParameter) = param.value
getvalue(param::HyperParameters) = broadcast(x->x.value,param.hyperparameters)
setvalue(param::HyperParameter,θ::Float64) = param.value=θ
setvalue(param::HyperParameters,θ::Float64) = broadcast(x->setvalue(x,θ),param.hyperparameters)
update!(param::HyperParameter,grad) = !param.fixed ? param.value += GradDescent.update(param.opt,grad) : nothing
update!(param::HyperParameters,grad) = for i in 1:length(param.hyperparameters); !param.hyperparameters[i].fixed ? param.hyperparameters[i].value += GradDescent.update(param.hyperparameters[i].opt,grad[i]):nothing;end;
isfixed(param::HyperParameter) = param.fixed


InnerProduct(X1,X2) = dot(X1,X2);
SquaredEuclidean(X1,X2) = norm(X1-X2,2)
Identity(X1,X2) = (X1,X2)



mutable struct KernelSum <: Kernel
    @kernelfunctionfields()
    kernel_array::Array{Kernel,1} #Array of summed kernels
    Nkernels::Int64
    #Constructors
    function KernelSum(kernels::AbstractArray)
        this = new("Sum of kernels")
        this.kernel_array = deepcopy(Array{Kernel,1}(kernels))
        this.Nkernels = length(this.kernel_array)
        this.pairwisefunction = Identity
        return this
    end
end
function compute(k::KernelSum,X1,X2,weight::Bool=true)
    sum = 0.0
    for kernel in k.kernel_array
        sum += compute(kernel,X1,X2)
    end
    return sum
end

function compute_deriv(k::KernelSum,X1,X2,weight::Bool=true)
    deriv_values = Array{Array{Any,1},1}()
    for kernel in k.kernel_array
        push!(deriv_values,compute_deriv(kernel,X1,X2,true))
    end
    return deriv_values
end

function compute_point_deriv(k::KernelSum,X1,X2)
    deriv_values = zeros(X1)
    for kernel in k.kernel_array
        deriv_values += getvalue(kernel.weight)*compute_point_deriv(kernel,X1,X2)
    end
    return deriv_values
end

function Base.:+(a::Kernel,b::Kernel)
    return KernelSum([a,b])
end
function Base.:+(a::KernelSum,b::Kernel)
    return KernelSum(vcat(a.kernel_array,b))
end
function Base.:+(a::KernelSum,b::KernelSum)
    return KernelSum(vcat(a.kernel_array,b.kernel_array))
end

function getindex(a::KernelSum,i::Int64)
    return a.kernel_array[i]
end

mutable struct KernelProduct <: Kernel
    @kernelfunctionfields
    kernel_array::Array{Kernel,1} #Array of multiplied kernels
    Nkernels::Int64
    function KernelProduct(kernels::AbstractArray)
        this = new("Product of kernels",HyperParameter(1.0))
        this.kernel_array = deepcopy(Array{Kernel,1}(kernels))
        this.Nkernels = length(this.kernel_array)
        this.pairwisefunction = Identity
        return this
    end
end

function compute(k::KernelProduct,X1,X2,weight::Bool=true)
    product = 1.0
    for kernel in k.kernel_array
        product *= compute(kernel,X1,X2,false)
    end
    return (weight?getvalue(k.weight):1.0)*product
end

function compute_deriv(k::KernelProduct,X1,X2,weight::Bool=true)
    tot = compute(k,X1,X2)
    deriv_values = Array{Array{Any,1},1}()
    for kernel in k.kernel_array
        push!(deriv_values,compute_deriv(kernel,X1,X2,false).*tot./compute(kernel,X1,X2,false))
    end
    if weight
        push!(deriv_values,[tot])
    end
    return deriv_values
end

function compute_point_deriv(k::KernelProduct,X1,X2)
    tot = compute(k,X1,X2)
    deriv_values = zeros(X1)
    for kernel in k.kernel_array
        #This should be checked TODO
        deriv_values += compute_point_deriv(kernel,X1,X2).*tot./compute(kernel,X1,X2)
    end
    return deriv_values
end

function Base.:*(a::Kernel,b::Kernel)
    return KernelProduct([a,b])
end
function Base.:*(a::KernelProduct,b::Kernel)
    return KernelProduct(vcat(a.kernel_array,b))
end
function Base.:*(a::KernelProduct,b::KernelSum)
    return KernelProduct(vcat(a.kernel_array,b.kernel_array))
end
function getindex(a::KernelProduct,i::Int64)
    return a.kernel_array[i]
end


"""
    Gaussian (RBF) Kernel
"""
mutable struct RBFKernel <: Kernel
    @kernelfunctionfields
    function RBFKernel(θ::Float64=1.0;coeff::Float64=1.0)
        return new("RBF",HyperParameter(coeff),HyperParameter(θ),1,SquaredEuclidean)
    end
end
function compute(k::RBFKernel,X1,X2,weight::Bool=true)
    if X1 == X2
      return (weight?getvalue(k.weight):1.0)
    end
    return (weight?getvalue(k.weight):1.0)*exp(-0.5*(k.pairwisefunction(X1,X2))^2/(getvalue(k.hyperparameters)^2))
end
#
function compute_deriv(k::RBFKernel,X1,X2,weight::Bool=true)
    a = k.pairwisefunction(X1,X2)
    if a != 0
        grad = a^2/(getvalue(k.hyperparameters)^3)*compute(k,X1,X2)
        if weight
            return [getvalue(k.weight)*grad,compute(k,X1,X2)]
        else
            return [grad]
        end
    else
      return [0.0]
    end
end

#TODO probably not right
function compute_point_deriv(k::RBFKernel,X1,X2)
    if X1 == X2
        return zeros(X1)
    else
        return getvalue(k.weight)*(-(X1-X2))./(getvalue(k.hyperparameters)^2).*compute(k,X1,X2)
    end
end

"""
    Laplace Kernel
"""

mutable struct LaplaceKernel <: Kernel
    @kernelfunctionfields
    function LaplaceKernel(θ::Float64=1.0;coeff::Float64=1.0)
        return new("Laplace",HyperParameter(coeff),HyperParameter(θ),1,SquaredEuclidean)
    end
end
function compute(k::LaplaceKernel,X1,X2,weight::Bool=true)
    if X1 == X2
      return (weight?getvalue(k.weight):1.0)
    end
    return (weight?getvalue(k.weight):1.0)*exp(-k.pairwisefunction(X1,X2)/(getvalue(k.hyperparameters)))
end
#
function compute_deriv(k::LaplaceKernel,X1,X2,weight::Bool=true)
    a = k.pairwisefunction(X1,X2)
    if a != 0
        grad = a/(getvalue(k.hyperparameters)^2)*compute(k,X1,X2)
        if weight
            return [getvalue(k.weight)*grad,compute(k,X1,X2)]
        else
            return [grad]
        end
    else
      return [0.0]
    end
end

#TODO Not correct
function compute_point_deriv(k::LaplaceKernel,X1,X2)
    if X1 == X2
        return zeros(X1)
    else
        return getvalue(k.weight)*(-(X1-X2))./(getvalue(k.hyperparameters)^2).*compute(k,X1,X2)
    end
end

"""
    Sigmoid Kernel
"""

mutable struct SigmoidKernel <: Kernel
    @kernelfunctionfields
    function SigmoidKernel(θ::Array{Float64}=[1.0,0.0];weight::Float64=1.0)
        return new("Sigmoid",HyperParameter(weight),HyperParameters(θ),length(θ),InnerProduct)
    end
end
function compute(k::SigmoidKernel,X1,X2,weight::Bool=true)
    return (weight?getvalue(k.weight):1.0)*tanh(getvalue(k.hyperparameters)[1]*k.pairwisefunction(X1,X2)+getvalue(k.hyperparameters)[2])
end
#
function compute_deriv(k::SigmoidKernel,X1,X2,weight::Bool=true)

    grad_1 = k.pairwisefunction(X1,X2)*(1-compute(k,X1,X2)^2)
    grad_2 = (1-compute(k,X1,X2)^2)
    if weight
        return [getvalue(k.weight)*grad_1,getvalue(k.weight)*grad_2,compute(k,X1,X2)]
    else
        return [grad_1,grad_2]
    end
end

function compute_point_deriv(k::SigmoidKernel,X1,X2)
    return getvalue(k.hyperparameters)[1]*X2.*(1-compute(k,X1,X2)^2)
end

"""
    Polynomial Kernel
"""

mutable struct PolynomialKernel <: Kernel
    @kernelfunctionfields
    function PolynomialKernel(θ::Array{Float64}=[1.0,0.0,2.0];weight::Float64=1.0)
        return new("Polynomial",HyperParameter(weight),HyperParameters(θ),length(θ),InnerProduct)
    end
end
function compute(k::PolynomialKernel,X1,X2,weight::Bool=true)
    return (weight?getvalue(k.weight):1.0)*(getvalue(k.hyperparameters)[1]*k.pairwisefunction(X1,X2)+getvalue(k.hyperparameters)[2])^getvalue(k.hyperparameters)[3]
end
#
function compute_deriv(k::PolynomialKernel,X1,X2,weight::Bool=true)
    grad_1 = getvalue(k.hyperparameters)[3]*k.pairwisefunction(X1,X2)*(getvalue(k.hyperparameters)[1]*k.pairwisefunction(X1,X2)+getvalue(k.hyperparameters)[2])^(getvalue(k.hyperparameters)[3]-1)
    grad_2 = getvalue(k.hyperparameters)[3]*(getvalue(k.hyperparameters)[1]*k.pairwisefunction(X1,X2)+getvalue(k.hyperparameters)[2])^(getvalue(k.hyperparameters)[3]-1)
    grad_3 = log(getvalue(k.hyperparameters)[1]*k.pairwisefunction(X1,X2)+getvalue(k.hyperparameters)[2])*compute(k,X1,X2)
    if weight
        return [getvalue(k.weight)*grad_1,getvalue(k.weight)*grad_2,getvalue(k.weight)*grad_3,compute(k,X1,X2)]
    else
        return [grad_1,grad_2,grad_3]
    end
end

function compute_point_deriv(k::PolynomialKernel,X1,X2)
    return getvalue(k.hyperparameters)[3]*getvalue(k.hyperparameters)[1]*X2*(getvalue(k.hyperparameters)[1]*k.pairwisefunction(X1,X2)+getvalue(k.hyperparameters)[2])^(getvalue(k.hyperparameters)[3]-1)
end

"""
    ARD Kernel
"""

mutable struct ARDKernel <: Kernel
    @kernelfunctionfields
    function ARDKernel(θ::Array{Float64}=[1.0];dim=0,weight::Float64=1.0)
        if length(θ)==1 && dim ==0
            error("You defined an ARD kernel without precising the number of dimensions
                             Please set dim in your kernel initialization or use ARDKernel(X,θ)")
        elseif dim!=0 && (length(params)!=dim && length(params)!=1)
            warn("You did not use the same dimension for your params and dim, using the first value of params for all dimensions")
            θ = ones(dim)*θ[1]
        elseif length(θ)==1 && dim!=0
            θ = ones(dim)*θ[1]
        end
        return new("ARD",HyperParameter(weight),HyperParameters(θ),length(θ),SquaredEuclidean)
    end
end
function compute(k::ARDKernel,X1,X2,weight::Bool=true)
    if X1==X2
        return 1.0
    end
    return (weight?getvalue(k.weight):1.0)*exp(-0.5*sum(((X1-X2)./getvalue(k.hyperparameters).^2)))
end
#
function compute_deriv(k::ARDKernel,X1,X2,weight::Bool=true)
    if X1 == X2
        return zeros(k.Nparameters)
    end
    return 2*(X1-X2).^2./(getvalue(k.hyperparameters).^3)*compute(k,X1,X2)
end

function compute_point_deriv(k::ARDKernel,X1,X2)
    if X1==X2
        return zeros(X1)
    end
    return -2*(X1-X2)./(getvalue(k.hyperparameters).^2).*compute(k,X1,X2)
end



# type Kernel
#     name::String #Type of function
#     kernel_function::Function # Kernel function
#     coeff::Float64 #Weight for the kernel
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
#                 warn("You did not use the same dimension for your params and dim, using the first value of params for all dimensions")
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
#         this.compute_deriv = function(X1,X2,weight)
#                 if weight
#                     return vcat(this.weight*this.derivative_kernel(X1,X2,this.param),this.kernel_function(X1,X2,this.param))
#                 else
#                     return this.derivative_kernel(X1,X2,this.param)
#                 end
#             end
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
function abel(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
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

function apply_gradients!(kernel::Kernel,gradients,weight::Bool=true)
    update!(kernel.hyperparameters,gradients[kernel.Nparameters ==1 ? 1 : 1:kernel.Nparameters])
    if weight
        update!(kernel.weight,gradients[end])
    end
end

function apply_gradients!(kernel::KernelSum,gradients,weight::Bool=true)
    for i in 1:kernel.Nkernels
        apply_gradients!(kernel.kernel_array[i],gradients[i],true)
    end
end

function apply_gradients!(kernel::KernelProduct,gradients,weight::Bool=true)
    for i in 1:kernel.Nkernels
        apply_gradients!(kernel.kernel_array[i],gradients[i],false);
    end
    if weight
        update!(kernel.weight,gradients[end][1]);
    end
end

function plotkernel(kernel::Kernel;range=[-3.0,3.0],npoints::Int64=100)
    if kernel.pairwisefunction == InnerProduct
        X1 = ones(npoints);
        X2 = collect(linspace(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.pairwisefunction == SquaredEuclidean
        X1 = zeros(npoints);
        X2 = collect(linspace(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.pairwisefunction == Identity
        plotlyjs()
        x = collect(linspace(range[1],range[2],npoints));
        value = broadcast((x,y)->compute(kernel,x,y),[i for i in x, j in x],[j for i in x, j in x])
        display(plot(x,x,value,t=:contour,fill=true,cbar=true,xlabel="X",ylabel="Y",title="k(X,Y)"))
    end
end

include("KernelMatrix.jl")

end #end of module
