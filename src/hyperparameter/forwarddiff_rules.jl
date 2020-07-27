function check_kernel_forward_diff(kernel)
    ps = Flux.params(kernel)
    isempty(ps) || return false
    p = first(ps)
    if p isa AbstractArray
        return true
    else
        @warn "ForwardDiff backend only works for simple kernels with `ARDTransform` or `ScaleTransform`, use `setadbackend(:reverse_diff)` to use reverse differentiation"
        return false
    end
end

function ∇L_ρ_forward(f, gp, X)
    check_kernel_forward_diff(kernel(gp))
    Jnn = kernelderivativematrix(kernel(gp), X)
    grads = map(f, Jnn)
    return IdDict{Any,Any}(first(Flux.params(kernel(gp))) => grads)
end

function ∇L_ρ_forward(f, gp::SparseVarLatent, X, ∇E_μ, ∇E_Σ, i, opt)
    Jmm = kernelderivativematrix(kernel(gp), gp.Z)
    Jnm = kernelderivativematrix(kernel(gp), X, gp.Z)
    Jnn = kerneldiagderivativematrix(kernel(gp), X)
    grads = f.(Jmm, Jnm, Jnn, Ref(∇E_μ), Ref(∇E_Σ), Ref(i), Ref(opt))
    return IdDict{Any,Any}(first(Flux.params(kernel(gp))) => grads)
end

function ∇L_ρ_forward(f, gp::OnlineVarLatent, X, ∇E_μ, ∇E_Σ, i, opt)
    Jmm = kernelderivativematrix(kernel(gp), gp.Z)
    Jnm = kernelderivativematrix(kernel(gp), X, gp.Z)
    Jnn = kerneldiagderivativematrix(kernel(gp), X)
    Jaa = kernelderivativematrix(kernel(gp), gp.Zₐ)
    Jab = kernelderivativematrix(kernel(gp), gp.Zₐ, gp.Z.Z)
    grads =
        map(f, Jmm, Jnm, Jnn, Jab, Jaa, Ref(∇E_μ), Ref(∇E_Σ), Ref(i), Ref(opt))
    return IdDict{Any,Any}(first(Flux.params(kernel(gp))) => grads)
end

## Return a function computing the gradient of the ELBO given the inducing point locations ##
function Z_gradient_forward(
    gp::SparseVarLatent{T},
    f_Z::Function,
    X,
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::Inference,
    opt::InferenceOptimizer,
) where {T<:Real}
    gradient_inducing_points = similar(gp.Z.Z)
    #preallocation
    Jmm, Jnm = indpoint_derivative(kernel(gp), gp.Z),
    indpoint_derivative(kernel(gp), X, gp.Z)
    for j = 1:gp.dim #Iterate over the points
        for k = 1:size(gp.Z, 2) #iterate over the dimensions
            @views gradient_inducing_points[j, k] =
                f_Z(Jmm[:, :, j, k], Jnm[:, :, j, k], ∇E_μ, ∇E_Σ, i, opt)
        end
    end
end

function Z_gradient_forward(
    gp::OnlineVarLatent{T},
    f_Z::Function,
    X,
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::Inference,
    opt::InferenceOptimizer,
) where {T<:Real}
    Z_gradient = similar(gp.Z.Z)
    Jnm, Jab, Jmm = indpoint_derivative(kernel(gp), X, gp.Z),
    indpoint_derivative(kernel(gp), gp.Zₐ, gp.Z),
    indpoint_derivative(kernel(gp), gp.Z)
    for j = 1:gp.dim #Iterate over the points
        for k = 1:size(gp.Z, 2) #iterate over the dimensions
            @views Z_gradient[j, k] = f_Z(
                Jmm[:, :, j, k],
                Jnm[:, :, j, k],
                Jab[:, :, j, k],
                ∇E_μ,
                ∇E_Σ,
                i,
                opt,
            )
        end
    end
    return Z_gradient
end

function indpoint_derivative(kernel::Kernel, Z::AbstractInducingPoints)
    reshape(
        ForwardDiff.jacobian(x -> kernelmatrix(kernel, x, obsdim = 1), Z),
        size(Z, 1),
        size(Z, 1),
        size(Z, 1),
        size(Z, 2),
    )
end

function indpoint_derivative(kernel::Kernel, X, Z::AbstractInducingPoints)
    reshape(
        ForwardDiff.jacobian(x -> kernelmatrix(kernel, X, x, obsdim = 1), Z),
        size(X, 1),
        size(Z, 1),
        size(Z, 1),
        size(Z, 2),
    )
end
