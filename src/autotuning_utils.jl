### Compute the gradients using a gradient function and matrices Js ###

base_kernel(k::Kernel) = eval(nameof(typeof(k)))


## VGP Case
function compute_hyperparameter_gradient(k::KernelWrapper,gradient_function::Function,J::Vector)
    return compute_hyperparameter_gradient.([k],[gradient_function],J)
end

function compute_hyperparameter_gradient(k::KernelWrapper,gradient_function::Function,J::AbstractMatrix)
    return gradient_function(J)
end

function compute_hyperparameter_gradient(k::KernelWrapper,gradient_function::Function,J::Nothing)
    return nothing
end

function compute_hyperparameter_gradient(k::KernelSumWrapper,gradient_function::Function,J::Vector)
    return [map(gradient_function,first(J)),compute_hyperparameter_gradient.(k,[gradient_function],J[end])]
end

function compute_hyperparameter_gradient(k::KernelProductWrapper,gradient_function::Function,J::Vector)
    return compute_hyperparameter_gradient.(k,J)
end

## SVGP Case
function compute_hyperparameter_gradient(k::KernelWrapper,gradient_function::Function,Jmm::Vector,Jnm::Vector,Jnn::Vector,∇E_μ::AbstractVector,∇E_Σ::AbstractVector,i::Inference,viopt::AbstractOptimizer)
    return compute_hyperparameter_gradient.([k],[gradient_function],Jmm,Jnm,Jnn,[∇E_μ],[∇E_Σ],i,[viopt])
end

function compute_hyperparameter_gradient(k::KernelWrapper,gradient_function::Function,Jmm::AbstractMatrix,Jnm::AbstractMatrix,Jnn::AbstractVector,∇E_μ::AbstractVector,∇E_Σ::AbstractVector,i::Inference,viopt::AbstractOptimizer)
    return gradient_function(Jmm,Jnm,Jnn,∇E_μ,∇E_Σ,i,viopt)
end

function compute_hyperparameter_gradient(k::KernelWrapper,gradient_function::Function,Jmm::Nothing,Jnm::Nothing,Jnn::Nothing,∇E_μ::AbstractVector,∇E_Σ::AbstractVector,i::Inference,viopt::AbstractOptimizer)
    return nothing
end

function compute_hyperparameter_gradient(k::KernelSumWrapper,gradient_function::Function,Jmm::Vector,Jnm::Vector,Jnn::Vector,∇E_μ::AbstractVector,∇E_Σ::AbstractVector,i::Inference,viopt::AbstractOptimizer)
    return [map(gradient_function,first(Jmm),first(Jnm),first(Jnn),[∇E_μ],[∇E_Σ],i,[viopt]),compute_hyperparameter_gradient.(k,[gradient_function],Jmm[end],Jnm[end],Jnn[end],[∇E_μ],[∇E_Σ],i,[viopt])]
end

function compute_hyperparameter_gradient(k::KernelProductWrapper,gradient_function::Function,Jmm::Vector,Jnm::Vector,Jnn::Vector,∇E_μ::AbstractVector,∇E_Σ::AbstractVector,i::Inference,viopt::AbstractOptimizer)
    return compute_hyperparameter_gradient.(k,Jmm,Jnm,Jnn,[∇E_μ],[∇E_Σ],i,[viopt])
end

##

function apply_gradients_lengthscale!(k::KernelWrapper,g::AbstractVector) where {T}
    ρ = params(k)
    newρ = []
    for i in 1:length(ρ)
        logρ = log.(ρ[i]) .+ update(k.opts[i],g[i].*ρ[i])
        push!(newρ,exp.(logρ))
    end
    set_params!(k,newρ)
end

function apply_gradients_lengthscale!(k::KernelWrapper{<:Kernel{T,<:ChainTransform}},g::AbstractVector) where {T}
    ρ = params(k)
    newρ = []
    ρt = first(ρ)
    gt = first(g)
    newρt = []
    for i in 1:length(ρt)
        if !isnothing(gt[i])
            if ρt[i] isa Real
                logρ = log(ρt[i]) + update(first(k.opts)[i],first(gt[i])*ρt[i])
                push!(newρt,exp(logρ))
            else
                logρ = log.(ρt[i]) .+ update(first(k.opts)[i],gt[i].*ρt[i])
                push!(newρt,exp.(logρ))
            end
        else
            push!(newρt,ρt[i])
        end
    end
    push!(newρ,newρt)
    if length(g) > 1
        for i in 2:length(ρ)
            logρ = log.(ρ[i]) .+ update(k.opts[i],g[i].*ρ[i])
            push!(newρ,exp.(logρ))
        end
    end
    KernelFunctions.set_params!(k,Tuple(newρ))
end

function apply_gradients_lengthscale!(k::KernelSumWrapper,g::AbstractVector)
    wgrads = first(g)
    w = k.weights
    logw = log.(w) + update(k.opt,w.*wgrads)
    k.weights .= exp.(w)
    apply_gradients_lengthscale!.(k,g[end])
end

function apply_gradients_lengthscale!(k::KernelProductWrapper,g::AbstractVector)
    apply_gradients_lengthscale!.(k,g)
end

function apply_gradients_variance!(gp::Abstract_GP,g::Real)
    logσ = log(gp.σ_k)
    logσ += update(gp.opt_σ,g*gp.σ_k)
    gp.σ_k = exp(logσ)
end

function apply_gradients_mean_prior!(μ::PriorMean,g::AbstractVector,X::AbstractMatrix)
    update!(μ,g,X)
end

## Wrapper for iterating over parameters for getting matrices

# Kernel Sum
function kernelderivative(kwrapper::KernelSumWrapper,X::AbstractMatrix)
    return [kernelmatrix.(kwrapper,[X]),kernelderivative.(kwrapper,[X])]
end

function kernelderivative(kwrapper::KernelSumWrapper,X::AbstractMatrix,Y::AbstractMatrix)
    return [kernelmatrix.(kwrapper,[X],[Y]),kernelderivative.(kwrapper,[X],[Y])]
end

function kerneldiagderivative(kwrapper::KernelSumWrapper,X::AbstractMatrix)
    return [kerneldiagmatrix.(kwrapper,[X]),kerneldiagderivative.(kwrapper,[X])]
end

# Kernel Product
recursive_hadamard(A::AbstractMatrix,V::AbstractVector) = recursive_hadamard.([A],V)
recursive_hadamard(A::AbstractMatrix,V::AbstractMatrix) = hadamard(A,V)
recursive_hadamard(A::AbstractVector,V::AbstractVector) = recursive_hadamard.([A],V)
recursive_hadamard(A::AbstractVector,V::AbstractVector{<:Real}) = hadamard(A,V)

function kernelderivative(kwrapper::KernelProductWrapper,X::AbstractMatrix)
    Kproduct = kernelmatrix(kwrapper)
    [recursive_hadamard([Kproduct./kernelmatrix(k,X)],kernelderivative(k,X)) for k in k.wrapper]
end

function kernelderivative(kwrapper::KernelProductWrapper,X::AbstractMatrix)
    Kproduct = kernelmatrix(kwrapper)
    [recursive_hadamard([Kproduct./kernelmatrix(k,X)],kernelderivative(k,X)) for k in k.wrapper]
end

function kerneldiagderivative(kwrapper::KernelProductWrapper,X::AbstractMatrix)
    Kproduct = kerneldiagmatrix(kwrapper)
    [recursive_hadamard([Kproduct./kerneldiagmatrix(k,X)],kerneldiagderivative(k,X)) for k in k.wrapper]
end

# Kernel
function kernelderivative(kernel::KernelWrapper,X::AbstractMatrix) where {T}
    ps = collect(KernelFunctions.opt_params(kernel.kernel))
    return [kernelderivative(kernel,ps,ps[i],i,X) for i in 1:length(ps)]
end

function kernelderivative(kernel::KernelWrapper,X::AbstractMatrix,Y::AbstractMatrix) where {T}
    ps = collect(KernelFunctions.opt_params(kernel.kernel))
    return [kernelderivative(kernel,ps,ps[i],i,X,Y) for i in 1:length(ps)]
end

function kerneldiagderivative(kernel::KernelWrapper,X::AbstractMatrix) where {T}
    ps = collect(KernelFunctions.opt_params(kernel.kernel))
    return [kerneldiagderivative(kernel,ps,ps[i],i,X) for i in 1:length(ps)]
end

## Take derivative of scalar hyperparameter ##
function kernelderivative(kernel::KernelWrapper,θ,θᵢ::Real,i::Int,X::AbstractMatrix)
    reshape(ForwardDiff.jacobian(x->begin
        newθ = [j==i ? first(x) : θ[j] for j in 1:length(θ)]
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1)
    end, [θᵢ])
        ,size(X,1),size(X,1))
end

function kernelderivative(kernel::KernelWrapper,θ,θᵢ::Real,i::Int,X::AbstractMatrix,Y::AbstractMatrix)
    reshape(ForwardDiff.jacobian(x->begin
        newθ = [j==i ? first(x) : θ[j] for j in 1:length(θ)]
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,Y,obsdim=1)
    end, [θᵢ])
        ,size(X,1),size(Y,1))
end

function kerneldiagderivative(kernel::KernelWrapper,θ,θᵢ::Real,i::Int,X::AbstractMatrix)
    reshape(ForwardDiff.jacobian(x->begin
        newθ = [j==i ? first(x) : θ[j] for j in 1:length(θ)]
        kerneldiagmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1)
    end, [θᵢ])
        ,size(X,1))
end

## Take derivative of vector hyperparameter ##
function kernelderivative(kernel::KernelWrapper,θ,θᵢ::AbstractVector{<:Real},i::Int,X::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ = [j==i ? x : θ[j] for j in 1:length(θ)] #Recreate a parameter vector
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1);
    end, θᵢ)
        ),size(X,1),size(X,1))
end

function kernelderivative(kernel::KernelWrapper,θ,θᵢ::AbstractVector{<:Real},i::Int,X::AbstractMatrix,Y::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ = [j==i ? x : θ[j] for j in 1:length(θ)] #Recreate a parameter vector
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,Y,obsdim=1);
    end, θᵢ)
        ),size(X,1),size(Y,1))
end

function kerneldiagderivative(kernel::KernelWrapper,θ,θᵢ::AbstractVector{<:Real},i::Int,X::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ = [j==i ? x : θ[j] for j in 1:length(θ)] #Recreate a parameter vector
        kerneldiagmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1);
    end, θᵢ)
        ),size(X,1))
end

## Take derivative of fixed hyperparameter (i.e. when transform is immutable)##
function kernelderivative(kernel::KernelWrapper,θ,θᵢ::Nothing,i::Int,X::AbstractMatrix)
    return nothing
end

function kernelderivative(kernel::KernelWrapper,θ,θᵢ::Nothing,i::Int,X::AbstractMatrix,Y::AbstractMatrix)
    return nothing
end

function kerneldiagderivative(kernel::KernelWrapper,θ,θᵢ::Nothing,i::Int,X::AbstractMatrix)
    return nothing
end

## Derivative of chain transform parameters ##
function kernelderivative(kernel::KernelWrapper,θ,θ_chaintransform::AbstractVector,i::Int,X::AbstractMatrix)
    return [kernelderivative(kernel,ps,ps[i],i,θ_chaintransform[j],j,X) for j in 1:length(θ_chaintransform)]
end

function kernelderivative(kernel::KernelWrapper,θ,θ_chaintransform::AbstractVector,i::Int,X::AbstractMatrix,Y::AbstractMatrix)
    return [kernelderivative(kernel,ps,ps[i],i,θ_chaintransform[j],j,X,Y) for j in 1:length(θ_chaintransform)]
end

function kerneldiagderivative(kernel::KernelWrapper,θ,θ_chaintransform::AbstractVector,i::Int,X::AbstractMatrix)
    return [kerneldiagderivative(kernel,ps,ps[i],i,θ_chaintransform[j],j,X) for j in 1:length(θ_chaintransform)]
end

## Derivative of chain transform parameters (Real) ##
function kernelderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::Real,j::Int,X::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ_t = [m==j ? first(x) : θ_t[m] for m in 1:length(θ_t)]
        newθ = vcat([newθ_t],θ[2:end])
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1)
    end, [θ_tj])
        ),size(X,1),size(X,1))
end

function kernelderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::Real,j::Int,X::AbstractMatrix,Y::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ_t = [m==j ? first(x) : θ_t[m] for m in 1:length(θ_t)]
        newθ = vcat([newθ_t],θ[2:end])
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,Y,obsdim=1)
    end, [θ_tj])
        ),size(X,1),size(Y,1))
end

function kerneldiagderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::Real,j::Int,X::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ_t = [m==j ? first(x) : θ_t[m] for m in 1:length(θ_t)]
        newθ = vcat([newθ_t],θ[2:end])
        kerneldiagmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1)
    end, [θ_tj])
        ),size(X,1))
end

## Derivative of chain transform parameters (Vector) ##
function kernelderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::AbstractVector,j::Int,X::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ_t = [m==j ? x : θ_t[m] for m in 1:length(θ_t)]
        newθ = vcat(newθ_t,θ[2:end])
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1);
    end, pt)
        ),size(X,1),size(X,1))
end

function kernelderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::AbstractVector,j::Int,X::AbstractMatrix,Y::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ_t = [m==j ? x : θ_t[m] for m in 1:length(θ_t)]
        newθ = vcat(newθ_t,θ[2:end])
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,Y,obsdim=1);
    end, pt)
        ),size(X,1),size(Y,1))
end

function kerneldiagderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::AbstractVector,j::Int,X::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ_t = [m==j ? x : θ_t[m] for m in 1:length(θ_t)]
        newθ = vcat(newθ_t,θ[2:end])
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1);
    end, pt)
        ),size(X,1))
end

## Derivative of chain transform parameters (immutable) ##
function kernelderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::Nothing,j::Int,X::AbstractMatrix)
    return nothing
end

function kernelderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::Nothing,j::Int,X::AbstractMatrix,Y::AbstractMatrix)
    return nothing
end

function kerneldiagderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::Nothing,j::Int,X::AbstractMatrix)
    return nothing
end

function indpoint_derivative(kernel::AbstractKernelWrapper,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,x,obsdim=1),Z),size(Z,1),size(Z,1),size(Z,1),size(Z,2))
end

function indpoint_derivative(kernel::AbstractKernelWrapper,X,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,X,x),Z),size(X,1),size(Z,1),size(Z,1),size(Z,2))
end
