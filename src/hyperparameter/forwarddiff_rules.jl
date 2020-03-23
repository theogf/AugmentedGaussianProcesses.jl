function ∇L_ρ_forward(f,gp,X)
    Jnn = kernelderivativematrix(gp.kernel,X,obsdim=1)
    grads = map(f,Jnn)
    return IdDict{Any,Any}(first(Flux.params(gp.kernel))=>grads)
end

function ∇L_ρ_forward(f,gp::_SVGP,X,∇E_μ,∇E_Σ,i,opt)
    Jmm = kernelderivativematrix(gp.kernel,gp.Z.Z,obsdim=1)
    Jnm = kernelderivativematrix(gp.kernel,X,gp.Z.Z,obsdim=1)
    Jnn = kerneldiagderivativematrix(gp.kernel,X,obsdim=1)
    grads = f.(Jmm,Jnm,Jnn,Ref(∇E_μ),Ref(∇E_Σ),Ref(i),Ref(opt))
    return IdDict{Any,Any}(first(Flux.params(gp.kernel))=>grads)
end

function ∇L_ρ_forward(f,gp::_OSVGP,X,∇E_μ,∇E_Σ,i,opt)
    Jmm = kernelderivativematrix(gp.kernel,gp.Z.Z,obsdim=1)
    Jnm = kernelderivativematrix(gp.kernel,X,gp.Z.Z,obsdim=1)
    Jnn = kerneldiagderivativematrix(gp.kernel,X,obsdim=1)
    Jaa = kernelderivativematrix(gp.kernel,gp.Zₐ,obsdim=1)
    Jab = kernelderivativematrix(gp.kernel,gp.Zₐ,gp.Z.Z,obsdim=1)
    grads = map(f,Jmm,Jnm,Jnn,Jab,Jaa,Ref(∇E_μ),Ref(∇E_Σ),Ref(i),Ref(opt))
    return IdDict{Any,Any}(first(Flux.params(gp.kernel))=>grads)
end

## Return a function computing the gradient of the ELBO given the inducing point locations ##
function Z_gradient_forward(gp::_SVGP{T},f_Z::Function,X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::InferenceOptimizer) where {T<:Real}
    gradient_inducing_points = similar(gp.Z.Z)
    #preallocation
    Jmm,Jnm = indpoint_derivative(gp.kernel,gp.Z),indpoint_derivative(gp.kernel,X,gp.Z)
    for j in 1:gp.dim #Iterate over the points
        for k in 1:size(gp.Z,2) #iterate over the dimensions
            @views gradient_inducing_points[j,k] = f_Z(Jmm[:,:,j,k],Jnm[:,:,j,k],∇E_μ,∇E_Σ,i,opt)
        end
    end
end

function Z_gradient_forward(gp::_OSVGP{T},f_Z::Function,X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::InferenceOptimizer) where {T<:Real}
    Z_gradient = similar(gp.Z.Z)
    Jnm,Jab,Jmm = indpoint_derivative(gp.kernel,X,gp.Z),indpoint_derivative(gp.kernel,gp.Zₐ,gp.Z), indpoint_derivative(gp.kernel,gp.Z)
    for j in 1:gp.dim #Iterate over the points
        for k in 1:size(gp.Z,2) #iterate over the dimensions
            @views Z_gradient[j,k] = f_Z(Jmm[:,:,j,k],Jnm[:,:,j,k],Jab[:,:,j,k],∇E_μ,∇E_Σ,i,opt)
        end
    end
    return Z_gradient
end



function kernelderivativematrix(kernel::Kernel,X::AbstractMatrix;obsdim=obsdim)
    global p = first(Flux.params(kernel))
    @assert p isa AbstractVector "ForwardDiff backend only works for simple kernels with ARD or ScaleTransform, use `setadbackend(:reverse_diff)` to use reverse differentiation"
    if length(p) == 1
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(first(x)),X,obsdim=obsdim),p),size(X,1),size(X,1),1)
    else
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(x),X,obsdim=obsdim),p),size(X,1),size(X,1),length(p))
    end
    return [J[:,:,i] for i in 1:length(p)]
end

function kernelderivativematrix(kernel::Kernel,X::AbstractMatrix,Y::AbstractMatrix;obsdim=obsdim)
    p = first(Flux.params(kernel))
    @assert p isa AbstractVector
    if length(p) == 1
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(first(x)),X,Y,obsdim=obsdim),p),size(X,1),size(Y,1),1)
    else
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(x),X,Y,obsdim=obsdim),p),size(X,1),size(Y,1),length(p))
    end
    return [J[:,:,i] for i in 1:length(p)]
end

function kerneldiagderivativematrix(kernel::Kernel,X::AbstractMatrix;obsdim=obsdim)
    p = first(Flux.params(kernel))
    @assert p isa AbstractVector
    if length(p) == 1
        J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(KernelFunctions.base_kernel(kernel)(first(x)),X,obsdim=obsdim),p),size(X,1),1)
    else
        J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(KernelFunctions.base_kernel(kernel)(x),X,obsdim=1),p),size(X,1),length(p))
    end
    return [J[:,i] for i in 1:length(p)]
end

function indpoint_derivative(kernel::Kernel,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,x,obsdim=1),Z),size(Z,1),size(Z,1),size(Z,1),size(Z,2))
end

function indpoint_derivative(kernel::Kernel,X,Z::InducingPoints)
    reshape(ForwardDiff.jacobian(x->kernelmatrix(kernel,X,x,obsdim=1),Z),size(X,1),size(Z,1),size(Z,1),size(Z,2))
end
