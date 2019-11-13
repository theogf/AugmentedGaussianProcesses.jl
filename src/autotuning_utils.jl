### Compute the gradients using a gradient function and matrices Js ###

base_kernel(k::Kernel) = eval(nameof(typeof(k)))

function compute_hyperparameter_gradient(k::Kernel,gradient_function::Function,J::Vector{<:AbstractMatrix})
    return map(gradient_function,J)
end

function compute_hyperparameter_gradient(k::Kernel,gradient_function::Function,Jmm::Vector{<:AbstractMatrix},Jnm::Vector{<:AbstractMatrix},Jnn::Vector{<:AbstractVector},∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::AbstractOptimizer) where {T<:Real}
    return map(gradient_function,Jmm,Jnm,Jnn,[∇E_μ],[∇E_Σ],i,[opt])
end


function apply_gradients_lengthscale!(opt::Optimizer,k::Kernel,g::AbstractVector)
    logρ = log.(KernelFunctions.get_params(k))
    logρ .+= update(opt,g.*exp.(logρ))
    set_params!(k,exp.(logρ))
end

function apply_gradients_variance!(gp::Abstract_GP,g::Real)
    logσ = log(gp.σ_k)
    logσ += update(gp.opt_σ,g*gp.σ_k)
    gp.σ_k = exp(logσ)
end

function apply_gradients_mean_prior!(μ::PriorMean,g::AbstractVector)
    update!(μ,g)
end

function kernelderivative(kernel::Kernel{T,<:ScaleTransform},X) where {T}
    p = KernelFunctions.get_params(kernel)
    J = reshape(ForwardDiff.jacobian(x->kernelmatrix(base_kernel(kernel)(x[1]),X,obsdim=1),p),size(X,1),size(X,1),length(p))
    return [J[:,:,i] for i in 1:length(p)]
end

function kernelderivative(kernel::Kernel{T,<:ScaleTransform},X,Y) where {T}
    p = KernelFunctions.get_params(kernel)
    J = reshape(ForwardDiff.jacobian(x->kernelmatrix(base_kernel(kernel)(x[1]),X,Y,obsdim=1),p),size(X,1),size(Y,1),length(p))
    return [J[:,:,i] for i in 1:length(p)]
end

function kerneldiagderivative(kernel::Kernel{T,<:ScaleTransform},X) where {T}
    p = KernelFunctions.get_params(kernel)
    J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(base_kernel(kernel)(x[1]),X,obsdim=1),p),size(X,1),length(p))
    return [J[:,i] for i in 1:length(p)]
end

function indpoint_derivative(kernel::Kernel,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,x,obsdim=1),Z.Z),size(Z,1),size(Z,1),size(Z,1),size(Z,2))
end

function indpoint_derivative(kernel::Kernel,X,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,X,x,obsdim=1),Z.Z),size(X,1),size(Z,1),size(Z,1),size(Z,2))
end
