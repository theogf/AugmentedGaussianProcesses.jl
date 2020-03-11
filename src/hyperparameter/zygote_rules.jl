function ∇L_ρ_reverse(f,gp,X)
    Zygote.gradient(()->_∇L_ρ_reverse(f,gp.kernel,X),Flux.params(gp.kernel)).grads # Zygote gradient
end

_∇L_ρ_reverse(f,kernel,X) = f(kernelmatrix(kernel,X,obsdim=1))

function ∇L_ρ_reverse(f,gp::_SVGP,X,∇E_μ,∇E_Σ,i,opt)
    Zygote.gradient(()->_∇L_ρ_reverse(f,gp.kernel,gp.Z.Z,X,∇E_μ,∇E_Σ,i,opt),Flux.params(gp.kernel)).grads # Zygote gradient
end

## Gradient ersatz for SVGP ##
function _∇L_ρ_reverse(f,kernel,Z,X,∇E_μ,∇E_Σ,i,opt)
    Kmm = kernelmatrix(kernel,Z,obsdim=1)
    Knm = kernelmatrix(kernel,X,Z,obsdim=1)
    Knn = diag(kernelmatrix(kernel,X,obsdim=1)) # TO FIX ONCE Zygote#429 is fixed.
    f(Kmm,Knm,Knn,∇E_μ,∇E_Σ,i,opt)
end

function ∇L_ρ_reverse(f,gp::_OSVGP,X,∇E_μ,∇E_Σ,i,opt)
    Zygote.gradient(()->_∇L_ρ_reverse(f,gp.kernel,gp.Z.Z,X,gp.Zₐ,∇E_μ,∇E_Σ,i,opt),Flux.params(gp.kernel)).grads # Zygote gradient
end

## Gradient ersatz for OSVGP ##
function _∇L_ρ_reverse(f,kernel,Z,X,Zₐ,∇E_μ,∇E_Σ,i,opt)
    Kmm = kernelmatrix(kernel,Z,obsdim=1)
    Knm = kernelmatrix(kernel,X,Z,obsdim=1)
    Knn = diag(kernelmatrix(kernel,X,obsdim=1)) # TO FIX ONCE Zygote#429 is fixed.
    Kaa = kernelmatrix(kernel,Zₐ,obsdim=1)
    Kab = kernelmatrix(kernel,Zₐ,Z,obsdim=1)
    f(Kmm,Knm,Knn,Kab,Kaa,∇E_μ,∇E_Σ,i,opt)
end

function Z_gradient_reverse(gp::_SVGP{T},f_Z::Function,X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::AbstractOptimizer) where {T<:Real}
    return Zygote.gradient(()->_Z_gradient_reverse(f_Z,gp.kernel,gp.Z.Z,X,∇E_μ,∇E_Σ,i,opt),Flux.params(gp.Z.Z)).grads[gp.Z.Z] # Zygote gradient
end

function _Z_gradient_reverse(f_Z,kernel,Z,X,∇E_μ,∇E_Σ,i,opt)
    Kmm = kernelmatrix(kernel,Z,obsdim=1)
    Knm = kernelmatrix(kernel,X,Z,obsdim=1)
    f_Z(Kmm,Knm,∇E_μ,∇E_Σ,i,opt)
end

function Z_gradient_reverse(gp::_OSVGP{T},f_Z::Function,X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::AbstractOptimizer) where {T<:Real}
    p = Flux.params(gp.Z.Z)
    return Zygote.gradient(()->_Z_gradient_reverse(f_Z,gp.kernel,gp.Z.Z,X,gp.Zₐ,∇E_μ,∇E_Σ,i,opt),p).grads[first(p)] # Zygote gradient
end

function _Z_gradient_reverse(f,kernel,Z,X,Zₐ,∇E_μ,∇E_Σ,i,opt)
    Kmm = kernelmatrix(kernel,Z,obsdim=1)
    Knm = kernelmatrix(kernel,X,Z,obsdim=1)
    Kab = kernelmatrix(kernel,Zₐ,Z,obsdim=1)
    f(Kmm,Knm,Kab,∇E_μ,∇E_Σ,i,opt)
end
