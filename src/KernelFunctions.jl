# """
#     Module for the kernel functions, also create kernel matrices
#         From the list http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
# """

# module KernelFunctions
macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end

using GradDescent
import Base: *, +
export Kernel
export CreateKernelMatrix, CreateDiagonalKernelMatrix
export delta_kroenecker

abstract type Kernel end;

abstract type AbstractHyperParameter end;

mutable struct HyperParameter <: AbstractHyperParameter
    value::Float64
    opt::Optimizer
    function HyperParameter(θ::Float64;opt::Optimizer=Adam())
        new(θ,opt)
    end
end

mutable struct HyperParameters <: AbstractHyperParameter
    hyperparameters::Array{HyperParameter,1}
end


getvalue(param::HyperParameter) = param.value
getvalue(param::HyperParameters) = broadcast(x->x.value,param.hyperparameters)
setvalue(param::HyperParameter,θ::Float64) = param.value=θ
setvalue(param::HyperParameters,θ::Float64) = broadcast(x->setvalue(x,θ),param.hyperparameters)
update!(param::HyperParameter,grad) = param.value += GradDescent.update(param.opt,grad)
# update!(param::HyperParameters,grad) = for i in 1:length(param.hyperparameters); param.hyperparameters[i].value += GradDescent.update(param.hyperparameters[i].opt,grad[i]);end;


@def kernelfunctionfields begin
    name::String #Name of the kernel function
    weight::HyperParameter #Weight of the kernel
    hyperparameters::AbstractHyperParameter #Parameters of the kernel
    Nparameters::Int64 #Number of parameters
end


mutable struct KernelSum <: Kernel
    @kernelfunctionfields
    kernel_array::Array{Kernel,1} #Array of summed kernels
    Nkernels::Int64
    #Constructors
    function KernelSum(kernels::Array{Kernel,1})
        this = new("Sum of kernels")
        this.kernel_array = deepcopy(kernels)
        this.Nkernels = length(this.kernel_array)
        return this
    end
end
function compute(k::KernelSum,X1,X2)
    sum = 0.0
    for kernel in k.kernel_array
        sum += getvalue(kernel.weight)*compute(kernel,X1,X2)
    end
    return sum
end

function compute_deriv(k::KernelSum,X1,X2,weight::Bool)
    deriv_values = Array{Array{Any,1},1}()
    for kernel in k.kernel_array
        push!(deriv_functions,compute_deriv(kernel,X1,X2,true))
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


mutable struct KernelProduct <: Kernel
    @kernelfunctionfields
    kernel_array::Array{Kernel,1} #Array of multiplied kernels
    Nkernels::Int64
    function KernelProduct(kernels::Array{Kernel,1})
        this = new("Product of kernels",HyperParameter(1.0))
        this.kernel_array = deepcopy(kernels)
        this.Nkernels = length(this.kernel_array)
        return this
    end
end

function compute(k::KernelProduct,X1,X2)
    product = 1.0
    for kernel in k.kernel_array
        product *= compute(kernel,X1,X2)
    end
    return product
end

function compute_deriv(k::KernelProduct,X1,X2,weight)
    tot = compute(k,X1,X2)
    deriv_values = Array{Array{Any,1},1}()
    for kernel in k.kernel_array
        push!(deriv_values,compute_deriv(kernel,X1,X2,weight=false).*tot./compute(kernel,X1,X2))
    end
    if weight
        push!(deriv_values,tot)
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
    return KernelProduct(Kernel[a,b])
end
function Base.:*(a::KernelSum,b::Kernel)
    return KernelProduct(vcat(a.kernel_array,b))
end
function Base.:*(a::KernelSum,b::KernelSum)
    return KernelProduct(vcat(a.kernel_array,b.kernel_array))
end


"""
    Gaussian (RBF) Kernel
"""
mutable struct RBFKernel <: Kernel
    @kernelfunctionfields
    function RBFKernel(θ::Float64=1.0)
        return new("RBF",HyperParameter(1.0),HyperParameter(θ),1)
    end
end
function compute(k::RBFKernel,X1,X2)
    if X1 == X2
      return 1
    end
    exp(-0.5*(norm(X1-X2))^2/(getvalue(k.hyperparameters)^2))
end
#
function compute_deriv(k::RBFKernel,X1,X2)
    a = norm(X1-X2)
    if a != 0
      return a^2/(getvalue(k.hyperparameters)^3)*rbf(k,X1,X2)
    else
      return 0
    end
end

function compute_point_deriv(k::RBFKernel,X1,X2)
    if X1 == X2
        return 0
    else
        return -(X1-X2)./(getvalue(k.hyperparameters)^2).*rbf(k,X1,X2)
    end
end

a= RBFKernel(3.0)
b= RBFKernel(0.5)
c = a+b
X1=rand(5)
X2=rand(5)
rc = compute(c,X1,X2)
ra = compute(a,X1,X2)
rb = compute(b,X1,X2)
diff = rc-(ra+rb)

d = a*b
d2 = a*b
e = d*d2
rd = compute(d,X1,X2)
diff = rd-ra*rb
re = compute(e,X1,X2)

ds
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
#         if kernel=="rbf"
#             this.kernel_function = rbf
#             this.derivative_kernel = deriv_rbf
#             this.derivative_point = deriv_point_rbf
#             this.param = params[1]
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
#         elseif kernel=="poly"
#             this.kernel_function = polynomial
#             this.derivative_kernel = deriv_polynomial
#             this.derivative_point = 0 # TODO
#             this.param = params
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



#Gaussian (RBF) Kernel
function rbf(X1,X2,Θ)
  if X1 == X2
    return 1
  end
  exp(-0.5*(norm(X1-X2))^2/(Θ[1]^2))
end


function deriv_rbf(X1,X2,θ)
  a = norm(X1-X2)
  if a != 0
    return norm(X1-X2)^2/(θ[1]^3)*rbf(X1,X2,θ)
  else
    return 0
  end
end

function deriv_point_rbf(X1,X2,θ)
    if X1 == X2
        return 0
    else
        return -(X1-X2)./(θ[1]^2).*rbf(X1,X2,θ)
    end
end


function ARD(X1::Array{Float64,1},X2::Array{Float64,1},θ)
    if X1==X2
        return 1
    end
    return exp(-0.5*sum(((X1-X2)./θ).^2))
end

function deriv_ARD(X1,X2,θ)
    if X1 == X2
        return 0
    end
    return 2*(X1-X2).^2./(θ.^3)*ARD(X1,X2,θ)
end

function deriv_point_ARD(X1,X2,θ)
    if X1==X2
        return 0
    end
    return -2*(X1-X2)./(θ.^2).*ARD(X1,X2,θ)
end

#Polynomial Kernel
function polynomial(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  (θ[1]*dot(X1,X2)+Θ[2])^θ[3]
end

function deriv_polynomial(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  grad_1 = θ[3]*dot(X1,X2)*(θ[1]*dot(X1,X2)+Θ[2])^(θ[3]-1)
  grad_2 = θ[3]*(θ[1]*dot(X1,X2)+Θ[2])^(θ[3]-1)
  grad_3 = log(θ[1]*dot(X1,X2)+Θ[2])*polynomial(X1,X2,θ)
  return [grad_1,grad_2,grad_3]
end

function deriv_point_polynomial(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    return θ[3]*θ[1]*X2*(θ[1]*dot(X1,X2)+Θ[2])^(θ[3]-1)
end

#Exponential Kernel
function exponential(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    return 0
end

function deriv_exponential(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    return 0
end

function deriv_point_exponential(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    return 0
end

#Laplace Kernel
function laplace(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    if X1 == X2
        return 1
    end
    return exp(-norm(X1-X2,2)/θ[1])
end

function deriv_laplace(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    a = norm(X1-X2,2)
    return a/(θ[1]^2)*exp(-a/θ[1])
end

function deriv_point_laplace(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    return 0
end

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

function sigmoid(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    return tanh(θ[1]*dot(X1,X2)+θ[2])
end

function deriv_sigmoid(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    grad_1 = dot(X1,X2)*(1-sigmoid(X1,X2,θ)^2)
    grad_2 = (1-sigmoid(X1,X2,θ)^2)
    return [grad_1,grad_2]
end

function deriv_point_sigmoid(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
    return θ[1]*X2.*(1-sigmoid(X1,X2,θ)^2)
end

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


function compute_kernel_gradients(kernel::Kernel,gradient_function::Function)
    gradients = gradient_function(kernel.compute_deriv)
    apply_gradients!(kernel,gradients)
end

function apply_gradients!(kernel::Kernel,gradients,weight)
    update!(kernel.param,gradients[1:kernel.Nparams])
    if weight
        update!(kernel.weight,gradients[end])
    end
end

function apply_gradients!(kernel::KernelSum,gradients,weight)
    for i in 1:kernel.Nkernels
        apply_gradients!(kernel[i],gradients[i],true)
    end
end

function apply_gradients!(kernel::KernelProduct,gradients,weight)
    for i in 1:kernel.Nkernels
        apply_gradients!(kernel[i],gradients[i],false)
    end
    if weight
        update!(kernel.weight,gradients[end])
    end
end

include("KernelMatrix.jl")

# end #end of module
