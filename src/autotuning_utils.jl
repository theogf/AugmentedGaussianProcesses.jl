### Compute the gradients using a gradient function and matrices Js ###

base_kernel(k::Kernel) = eval(nameof(typeof(k)))

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

function compute_hyperparameter_gradient(k::Kernel,gradient_function::Function,Jmm::Vector{<:AbstractMatrix},Jnm::Vector{<:AbstractMatrix},Jnn::Vector{<:AbstractVector},∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::AbstractOptimizer) where {T<:Real}
    return map(gradient_function,Jmm,Jnm,Jnn,[∇E_μ],[∇E_Σ],i,[opt])
end

function apply_gradients_lengthscale!(k::KernelWrapper,g::AbstractVector) where {T}
    ρ = params(k)
    newρ = []
    for i in 1:length(ρ)
        logρ = log.(ρ[i]) .+ update(k.opts[i],g[i].*ρ[i])
        @info logρ
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

function kernelderivative(kwrapper::KernelSumWrapper,X::AbstractMatrix)
    return [kernelmatrix.(kwrapper,[X]),kernelderivative.(kwrapper,[X])]
end

function kernelderivative(kwrapper::KernelProductWrapper,X::AbstractMatrix)
    Kproduct = kernelmatrix(kwrapper)
    [hadamard.(kernelderivative(k,X),[Kproduct./kernelmatrix(k,X)]) for k in k.wrapper] ### TO CORRECT!!!!
end

function kernelderivative(kernel::KernelWrapper,X::AbstractMatrix) where {T}
    ps = collect(KernelFunctions.opt_params(kernel.kernel))
    return [kernelderivative(kernel,ps,ps[i],i,X) for i in 1:length(ps)]
end

## Take derivative of scalar hyperparameter ##
function kernelderivative(kernel::KernelWrapper,θ,θᵢ::Real,i::Int,X::AbstractMatrix)
    reshape.(eachcol(
        ForwardDiff.jacobian(x->begin
        newθ = [j==i ? first(x) : θ[j] for j in 1:length(θ)]
        kernelmatrix(KernelFunctions.duplicate(kernel.kernel,newθ),X,obsdim=1)
    end, [θᵢ])
        ),size(X,1),size(X,1))
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

## Take derivative of fixed hyperparameter (i.e. when transform is immutable)##
function kernelderivative(kernel::KernelWrapper,θ,θᵢ::Nothing,i::Int,X::AbstractMatrix)
    return nothing
end

## Derivative of chain transform parameters ##
function kernelderivative(kernel::KernelWrapper,θ,θ_chaintransform::AbstractVector,i::Int,X::AbstractMatrix)
    return [kernelderivative(kernel,ps,ps[i],i,θ_chaintransform[j],j,X) for j in 1:length(θ_chaintransform)]
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

## Derivative of chain transform parameters (immutable) ##
function kernelderivative(kernel::KernelWrapper,θ,θ_t::AbstractVector,i::Int,θ_tj::Nothing,j::Int,X::AbstractMatrix)
    return nothing
end


function kernelderivative(kernel::Kernel{T,<:ScaleTransform{<:Base.RefValue}},X,Y) where {T}
    p = collect(KernelFunctions.params(kernel))
    J = reshape(ForwardDiff.jacobian(x->kernelmatrix(base_kernel(kernel)(x...),X,Y,obsdim=1),p),size(X,1),size(Y,1),1)
    return [J[:,:,1]]
end

function kernelderivative(kernel::Kernel{T,<:ScaleTransform{<:AbstractVector}},X,Y) where {T}
    p = first(KernelFunctions.params(kernel))
    J = reshape(ForwardDiff.jacobian(x->kernelmatrix(base_kernel(kernel)(x),X,Y,obsdim=1),p),size(X,1),size(Y,1),length(p))
    return [J[:,:,i] for i in 1:length(p)]
end

function kerneldiagderivative(kernel::Kernel{T,<:ScaleTransform{<:Base.RefValue}},X) where {T}
    p = collect(KernelFunctions.params(kernel))
    J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(base_kernel(kernel)(x...),X,obsdim=1),p),size(X,1),length(p))
    return [J[:,1]]
end

function kerneldiagderivative(kernel::Kernel{T,<:ScaleTransform{<:AbstractVector}},X) where {T}
    p = first(KernelFunctions.params(kernel))
    J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(base_kernel(kernel)(x),X,obsdim=1),p),size(X,1),length(p))
    return [J[:,i] for i in 1:length(p)]
end

function indpoint_derivative(kernel::AbstractKernelWrapper,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,x,obsdim=1),Z),size(Z,1),size(Z,1),size(Z,1),size(Z,2))
end

function indpoint_derivative(kernel::AbstractKernelWrapper,X,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,X,x),Z),size(X,1),size(Z,1),size(Z,1),size(Z,2))
end
