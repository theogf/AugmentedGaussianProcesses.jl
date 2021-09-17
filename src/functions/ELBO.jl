function ELBO(model::GP, X, y, pr_mean, kernel)
    setpr_mean!(model.f, pr_mean)
    setkernel!(model.f, kernel)
    state = compute_kernel_matrices(model, (;), X, true)
    return objective(model, state, y)
end

@traitfn function ELBO(
    model::TGP, X, y, pr_means, kernels
) where {TGP <: AbstractGPModel; IsFull{TGP}}
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    state = compute_kernel_matrices!(model, (;), X, true)
    return objective(model, state, y)
end

@traitfn function ELBO(
    model::TGP, X, y, pr_means, kernels, Zs
) where {TGP <: AbstractGPModel; !IsFull{TGP}}
    setpr_means!(model, pr_means)
    setkernels!(model, kernels)
    setZs!(model, Zs)
    state = compute_kernel_matrices(model, (;), X, true)
    return objective(model, state, y)
end
