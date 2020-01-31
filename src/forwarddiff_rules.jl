function ∇L_ρ_forward(f,gp,X)
    Jnn = first(gp.σ_k)*kernelderivativematrix(gp.kernel,X,obsdim=1)
    grads = map(f,Jnn)
    return IdDict{Any,Any}(first(KernelFunctions.params(gp.kernel))=>grads)
end

function ∇L_ρ_forward(f,gp::_SVGP,X,∇E_μ,∇E_Σ,i,opt)
    Jmm = first(gp.σ_k).*kernelderivativematrix(gp.kernel,gp.Z.Z,obsdim=1)
    Jnm = first(gp.σ_k).*kernelderivativematrix(gp.kernel,X,gp.Z.Z,obsdim=1)
    Jnn = first(gp.σ_k).*kerneldiagderivativematrix(gp.kernel,X,obsdim=1)
    grads = f.(Jmm,Jnm,Jnn,Ref(∇E_μ),Ref(∇E_Σ),Ref(i),Ref(opt))
    return IdDict{Any,Any}(first(KernelFunctions.params(gp.kernel))=>grads)
end

function ∇L_ρ_forward(f,gp::_OSVGP,X,∇E_μ,∇E_Σ,i,opt)
    Jmm = first(gp.σ_k)*kernelderivativematrix(gp.kernel,gp.Z.Z,obsdim=1)
    Jnm = first(gp.σ_k)*kernelderivativematrix(gp.kernel,X,gp.Z.Z,obsdim=1)
    Jnn = first(gp.σ_k)*kerneldiagderivativematrix(gp.kernel,X,obsdim=1)
    Jaa = first(gp.σ_k)*kernelderivativematrix(kernel,gp.Zₐ,obsdim=1)
    Jab = first(gp.σ_k)*kernelderivativematrix(kernel,gp.Zₐ,gp.Z.Z,obsdim=1)
    grads = map(f,Jmm,Jnm,Jnn,Jaa,Jab,Ref(∇E_μ),Ref(∇E_Σ),Ref(i),Ref(opt))
    return IdDict{Any,Any}(first(KernelFunctions.params(gp.kernel))=>grads)
end

function kernelderivativematrix(kernel::Kernel,X::AbstractMatrix;obsdim=obsdim)
    global p = first(KernelFunctions.params(kernel))
    @assert p isa AbstractVector "ForwardDiff backend only works for simple kernels with ARD or ScaleTransform"
    if length(p) == 1
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(first(x)),X,obsdim=obsdim),p),size(X,1),size(X,1),1)
    else
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(x),X,obsdim=obsdim),p),size(X,1),size(X,1),length(p))
    end
    return [J[:,:,i] for i in 1:length(p)]
end

function kernelderivativematrix(kernel::Kernel,X::AbstractMatrix,Y::AbstractMatrix;obsdim=obsdim)
    p = first(KernelFunctions.params(kernel))
    @assert p isa AbstractVector
    if length(p) == 1
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(first(x)),X,Y,obsdim=obsdim),p),size(X,1),size(Y,1),1)
    else
        J = reshape(ForwardDiff.jacobian(x->kernelmatrix(KernelFunctions.base_kernel(kernel)(x),X,Y,obsdim=obsdim),p),size(X,1),size(Y,1),length(p))
    end
    return [J[:,:,i] for i in 1:length(p)]
end

function kerneldiagderivativematrix(kernel::Kernel,X::AbstractMatrix;obsdim=obsdim)
    p = first(KernelFunctions.params(kernel))
    @assert p isa AbstractVector
    if length(p) == 1
        J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(KernelFunctions.base_kernel(kernel)(first(x)),X,obsdim=obsdim),p),size(X,1),1)
    else
        J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(KernelFunctions.base_kernel(kernel)(x),X,obsdim=1),p),size(X,1),length(p))
    end
    return [J[:,i] for i in 1:length(p)]
end
