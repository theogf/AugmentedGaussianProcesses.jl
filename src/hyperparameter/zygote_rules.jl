function ∇L_ρ_zygote(f, gp::AbstractLatent, X)
    k = kernel(gp)
    return (Zygote.gradient(params(k)) do
        _∇L_ρ_zygote(f, k, X)
    end).grads # Zygote gradient
end

_∇L_ρ_zygote(f, k, X) = f(kernelmatrix(k, X))

function ∇L_ρ_zygote(f, gp::SparseVarLatent, X, ∇E_μ, ∇E_Σ, i, opt)
    k = kernel(gp)
    return (Zygote.gradient(params(k)) do
        _∇L_ρ_zygote(f, k, gp.Z, X, ∇E_μ, ∇E_Σ, i, opt)
    end).grads
end

## Gradient ersatz for SVGP ##
function _∇L_ρ_zygote(f, kernel, Z, X, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z)
    Knm = kernelmatrix(kernel, X, Z)
    Knn = diag(kernelmatrix(kernel, X)) # TO FIX ONCE Zygote#429 is fixed.
    f(Kmm, Knm, Knn, ∇E_μ, ∇E_Σ, i, opt)
end

function ∇L_ρ_zygote(f, gp::OnlineVarLatent, X, ∇E_μ, ∇E_Σ, i, opt)
    k = kernel(gp)
    Zrv = RowVecs(copy(hcat(gp.Z...)')) # TODO Fix that once https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/151 is solved
    Zarv = RowVecs(copy(hcat(gp.Zₐ...)'))
    return  (Zygote.gradient(params(k)) do
        _∇L_ρ_zygote(f, k, Zrv, X, Zarv, ∇E_μ, ∇E_Σ, i, opt)
    end).grads
end

## Gradient ersatz for OSVGP ##
function _∇L_ρ_zygote(f, kernel, Z, X, Zₐ, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z)
    Knm = kernelmatrix(kernel, X, Z)
    Knn = diag(kernelmatrix(kernel, X)) # Workaround
    Kaa = kernelmatrix(kernel, Zₐ)
    Kab = kernelmatrix(kernel, Zₐ, Z)
    f(Kmm, Knm, Knn, Kab, Kaa, ∇E_μ, ∇E_Σ, i, opt)
end

function Z_gradient_zygote(
    gp::SparseVarLatent{T},
    f_Z::Function,
    X,
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::Inference,
    opt::InferenceOptimizer,
) where {T<:Real}
    p = params(gp.Z)
    return (Zygote.gradient(p) do
            _Z_gradient_zygote(f_Z, kernel(gp), gp.Z, X, ∇E_μ, ∇E_Σ, i, opt)
        end).grads[first(p)]
end

function _Z_gradient_zygote(f_Z, kernel, Z, X, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z)
    Knm = kernelmatrix(kernel, X, Z)
    f_Z(Kmm, Knm, ∇E_μ, ∇E_Σ, i, opt)
end

function Z_gradient_zygote(gp::OnlineVarLatent{T},f_Z::Function,X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::InferenceOptimizer) where {T<:Real}
    p = params(gp.Z)
    return (Zygote.gradient(p) do
        _Z_gradient_zygote(f_Z,kernel(gp),gp.Z.Z,X,gp.Zₐ,∇E_μ,∇E_Σ,i,opt)
    end).grads[first(p)] # Zygote gradient
end

function _Z_gradient_zygote(f, kernel, Z, X, Zₐ, ∇E_μ, ∇E_Σ, i, opt)
    Kmm = kernelmatrix(kernel, Z)
    Knm = kernelmatrix(kernel, X, Z)
    Kab = kernelmatrix(kernel, Zₐ, Z)
    f(Kmm, Knm, Kab, ∇E_μ, ∇E_Σ, i, opt)
end
