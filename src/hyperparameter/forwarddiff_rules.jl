function ∇L_ρ_forward(f, gp::AbstractLatent, X::AbstractVector)
    θ, re = destructure(kernel(gp))
    return g = ForwardDiff.gradient(θ) do x
        k = re(x)
        Knn = kernelmatrix(k, X)
        f(Knn)
    end
end

function ∇L_ρ_forward(f, gp::SparseVarLatent, X::AbstractVector, ∇E_μ, ∇E_Σ, i, opt)
    θ, re = destructure(kernel(gp))
    return g = ForwardDiff.gradient(θ) do x
        k = re(x)
        Kmm = kernelmatrix(k, Zview(gp))
        Knm = kernelmatrix(k, X, Zview(gp))
        Knn = kernelmatrix_diag(k, X)
        f(Kmm, Knm, Knn, ∇E_μ, ∇E_Σ, i, opt)
    end
end

function ∇L_ρ_forward(f, gp::OnlineVarLatent, X::AbstractVector, ∇E_μ, ∇E_Σ, i, opt)
    θ, re = destructure(kernel(gp))
    return g = ForwardDiff.gradient(θ) do x
        k = re(x)
        Kmm = kernelmatrix(k, Zview(gp))
        Knm = kernelmatrix(k, X, Zview(gp))
        Knn = kernelmatrix_diag(k, X)
        Kaa = kernelmatrix(k, gp.Zₐ)
        Kab = kernelmatrix(k, gp.Zₐ, Zview(gp))
        f(Kmm, Knm, Knn, Kab, Kaa, ∇E_μ, ∇E_Σ, i, opt)
    end
end

## Return a function computing the gradient of the ELBO given the inducing point locations ##
function Z_gradient_forward(
    gp::SparseVarLatent{T},
    f_Z::Function,
    X,
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::AbstractInference,
    opt::InferenceOptimizer,
) where {T<:Real}
    gradient_inducing_points = similar(gp.Z.Z)
    #pre-allocation
    Jmm, Jnm = indpoint_derivative(kernel(gp), gp.Z),
    indpoint_derivative(kernel(gp), X, gp.Z)
    for j in 1:(gp.dim) #Iterate over the points
        for k in 1:size(gp.Z, 2) #iterate over the dimensions
            @views gradient_inducing_points[j, k] .= f_Z(
                Jmm[:, :, j, k], Jnm[:, :, j, k], ∇E_μ, ∇E_Σ, i, opt
            )
        end
    end
end

function Z_gradient_forward(
    gp::OnlineVarLatent{T},
    f_Z::Function,
    X,
    ∇E_μ::AbstractVector{T},
    ∇E_Σ::AbstractVector{T},
    i::AbstractInference,
    opt::InferenceOptimizer,
) where {T<:Real}
    Z_gradient = similar(gp.Z.Z)
    Jnm, Jab, Jmm = indpoint_derivative(kernel(gp), X, gp.Z),
    indpoint_derivative(kernel(gp), gp.Zₐ, gp.Z),
    indpoint_derivative(kernel(gp), gp.Z)
    for j in 1:(gp.dim) #Iterate over the points
        for k in 1:size(gp.Z, 2) #iterate over the dimensions
            @views Z_gradient[j, k] .= f_Z(
                Jmm[:, :, j, k], Jnm[:, :, j, k], Jab[:, :, j, k], ∇E_μ, ∇E_Σ, i, opt
            )
        end
    end
    return Z_gradient
end

function indpoint_derivative(kernel::Kernel, Z::AbstractVector)
    return reshape(
        ForwardDiff.jacobian(x -> kernelmatrix(kernel, x; obsdim=1), Z),
        size(Z, 1),
        size(Z, 1),
        size(Z, 1),
        size(Z, 2),
    )
end

function indpoint_derivative(kernel::Kernel, X, Z::AbstractVector)
    return reshape(
        ForwardDiff.jacobian(x -> kernelmatrix(kernel, X, x; obsdim=1), Z),
        size(X, 1),
        size(Z, 1),
        size(Z, 1),
        size(Z, 2),
    )
end
