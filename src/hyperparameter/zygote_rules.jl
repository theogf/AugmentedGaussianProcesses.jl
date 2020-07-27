function ∇L_ρ_reverse(f, gp::AbstractLatent, X)
    k = kernel(gp)
    return (Flux.gradient(Flux.params(k)) do
        _∇L_ρ_reverse(f, k, X)
    end).grads # Zygote gradient
end

_∇L_ρ_reverse(f, k, X) = f(kernelmatrix(k, X, obsdim = 1))

function ∇L_ρ_reverse(f, gp::SparseVarLatent, X, ∇E_μ, ∇E_Σ, i, opt)
    k = kernel(gp)
    return (Zygote.gradient(Flux.params(k)) do
        _∇L_ρ_reverse(f, k, gp.Z, X, ∇E_μ, ∇E_Σ, i, opt)
    end).grads
end

## Gradient ersatz for SVGP ##
function _∇L_ρ_reverse(f, kernel, Z, X, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z, obsdim = 1)
    Knm = kernelmatrix(kernel, X, Z, obsdim = 1)
    Knn = diag(kernelmatrix(kernel, X, obsdim = 1)) # TO FIX ONCE Zygote#429 is fixed.
    f(Kmm, Knm, Knn, ∇E_μ, ∇E_Σ, i, opt)
end

function ∇L_ρ_reverse(f, gp::OnlineVarLatent, X, ∇E_μ, ∇E_Σ, i, opt)
    k = kernel(gp)
    return  (Zygote.gradient(Flux.params(k)) do
        _∇L_ρ_reverse(f, k, gp.Z.Z, X, gp.Zₐ, ∇E_μ, ∇E_Σ, i, opt)
    end).grads
end

## Gradient ersatz for OSVGP ##
function _∇L_ρ_reverse(f, kernel, Z, X, Zₐ, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z)
    Knm = kernelmatrix(kernel, X, Z)
    Knn = kerneldiagmatrix(kernel, X)
    Kaa = kernelmatrix(kernel, Zₐ)
    Kab = kernelmatrix(kernel, Zₐ, Z)
    f(Kmm, Knm, Knn, Kab, Kaa, ∇E_μ, ∇E_Σ, i, opt)
end

function Z_gradient_reverse(
    gp::SparseVarLatent{T},
    f_Z::Function,
    X,
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::Inference,
    opt::InferenceOptimizer,
) where {T<:Real}
    p = Flux.params(gp.Z)
    return (Zygote.gradient(p) do
            _Z_gradient_reverse(f_Z, kernel(gp), gp.Z, X, ∇E_μ, ∇E_Σ, i, opt)
        end).grads[first(p)]
end

function _Z_gradient_reverse(f_Z, kernel, Z, X, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z)
    Knm = kernelmatrix(kernel, X, Z)
    f_Z(Kmm, Knm, ∇E_μ, ∇E_Σ, i, opt)
end

function Z_gradient_reverse(gp::OnlineVarLatent{T},f_Z::Function,X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::InferenceOptimizer) where {T<:Real}
    p = Flux.params(gp.Z)
    return (Zygote.gradient(p) do
        _Z_gradient_reverse(f_Z,kernel(gp),gp.Z.Z,X,gp.Zₐ,∇E_μ,∇E_Σ,i,opt)
    end).grads[first(p)] # Zygote gradient
end

function _Z_gradient_reverse(f, kernel, Z, X, Zₐ, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z)
    Knm = kernelmatrix(kernel, X, Z)
    Kab = kernelmatrix(kernel, Zₐ, Z)
    f(Kmm, Knm, Kab, ∇E_μ, ∇E_Σ, i, opt)
end
